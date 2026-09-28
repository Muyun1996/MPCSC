# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from dataclasses import dataclass, field
from typing import Optional
from typing import List

import gym
import numpy as np
import omegaconf
from omegaconf import OmegaConf, ListConfig
import torch

import mbrl.constants
import mbrl.models
import mbrl.planning
import mbrl.types
import mbrl.util
import mbrl.util.common
import mbrl.util.math

import datetime
import platform
import wandb
import cProfile

EVAL_LOG_FORMAT = mbrl.constants.EVAL_LOG_FORMAT

def train(
    env: gym.Env,
    termination_fn: mbrl.types.TermFnType,
    reward_fn: mbrl.types.RewardFnType,
    cfg: omegaconf.DictConfig,
    silent: bool = False,
    work_dir: Optional[str] = None,
) -> np.float32:
    # ------------------- init wandb-------------------
    sys_platform = platform.platform().lower()
    if "linux" in sys_platform:
        OmegaConf.update(cfg, "system", "linux", force_add=True)
    elif "darwin" in sys_platform:
        OmegaConf.update(cfg, "system", "mac", force_add=True)
    elif "windows" in sys_platform:
        OmegaConf.update(cfg, "system", "windows", force_add=True)
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("-%m-%d %H:%M")
    wandb.init(
        # set the wandb project where this run will be logged
        project="mbrl_pets",
        entity="moumuyun",
        name=cfg.name + formatted_time,
        # track hyperparameters and run metadata
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    OmegaConf.save(config=cfg, f=os.path.join(wandb.run.dir, 'conf.yaml'))

    # ------------------- Initialization -------------------
    debug_mode = cfg.get("debug_mode", False)

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    work_dir = work_dir or os.getcwd()
    print(f"Results will be saved at {work_dir}.")

    if silent:
        logger = None
    else:
        logger = mbrl.util.Logger(work_dir)
        logger.register_group(
            mbrl.constants.RESULTS_LOG_NAME, EVAL_LOG_FORMAT, color="green"
        )

    # -------- add driver_node and adj_mat to cfg ----------
    @dataclass
    class Network_data:
        node_num: int = 0
        driver_node_num: int = 0
        driver_node: ListConfig = field(default_factory=lambda: OmegaConf.create([]))
        has_adj_mat: bool = False
        adj_mat: ListConfig = field(default_factory=lambda: OmegaConf.create([]))
        use_node_id: bool = False

    network_config: Network_data = OmegaConf.structured(Network_data)
    driver_node = [int(x) for x in env.unwrapped.driver_node]
    network_config.node_num = env.unwrapped.node_num
    network_config.driver_node_num = env.unwrapped.driver_node_num
    network_config.driver_node = driver_node
    if hasattr(cfg.overrides, "use_node_id"):
        network_config.use_node_id = cfg.overrides.use_node_id
    if hasattr(env.unwrapped, 'adj_mat'):
        network_config.has_adj_mat = True
        network_config.adj_mat = [[int(x) for x in row] for row in env.unwrapped.adj_mat]
    else:
        network_config.has_adj_mat = False
        network_config.adj_mat = [[]]
    cfg.dynamics_model = OmegaConf.merge(cfg.dynamics_model, network_config)

    # -------- Create and populate initial env dataset --------
    dynamics_model = mbrl.util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape)
    use_double_dtype = cfg.algorithm.get("normalize_double_precision", False)
    dtype = np.double if use_double_dtype else np.float32
    replay_buffer = mbrl.util.common.create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        rng=rng,
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
    )
    replay_buffer_test = mbrl.util.common.create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        rng=rng,
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
    )
    mbrl.util.common.rollout_agent_trajectories(
        env,
        cfg.overrides.is_random_init,
        cfg.algorithm.initial_obs_steps,
        mbrl.planning.DefaultAgent(env),
        {},
        replay_buffer=replay_buffer,
        work_dir=work_dir,
        )
    replay_buffer.save(work_dir, "replay_buffer_offline")
    replay_buffer.save(work_dir, "replay_buffer")
    mbrl.util.common.rollout_agent_trajectories_for_test(
        env,
        cfg.overrides.is_random_init,
        cfg.algorithm.test_steps,
        mbrl.planning.RandomAgent(env),
        {},
        replay_buffer=replay_buffer_test,
        work_dir=work_dir,
    )
    replay_buffer_test.save(work_dir, "replay_buffer_test")



    # ---------------------------------------------------------
    # ---------- Create model environment and agent -----------
    model_env = mbrl.models.ModelEnv(
        env, dynamics_model, termination_fn, reward_fn, generator=torch_generator, epistemic_reward_weight=cfg.algorithm.epistemic_reward_weight, action_cost_reward_weight=cfg.algorithm.action_cost_reward_weight
    )
    model_trainer = mbrl.models.ModelTrainer(
        dynamics_model,
        optim_lr=cfg.overrides.model_lr,
        weight_decay=cfg.overrides.model_wd,
        logger=logger,
    )

    agent = mbrl.planning.create_trajectory_optim_agent_for_model(
        model_env, cfg.algorithm.agent, num_particles=cfg.algorithm.num_particles
    )
    # ---------------------------------------------------------
    # --------------------- Training Loop ---------------------
    env_steps = 0
    current_trial = 0
    max_total_reward = -np.inf
    default_num_epochs_train_model = cfg.overrides.num_epochs_train_model
    while env_steps < cfg.overrides.num_steps:
        obs = env.reset(random_init=False)
        agent.reset()
        done = False
        total_cost_reward = 0.0
        total_reward = 0.0
        steps_trial = 0

        # Reduce the epistemic_reward_weight as agent interact with the real environment
        if env_steps > 4 * cfg.overrides.trial_length:
            model_env.epistemic_reward_weight = 0
        # log metrics to wandb
        wandb.log({"epistemic_reward_weight": model_env.epistemic_reward_weight})

        while not done:
            # --------------- Model Training -----------------
            # TODO: check
            if env_steps == 0:
                train_using_only_offline_data = True
                cfg.overrides.num_epochs_train_model = cfg.overrides.init_num_epochs_train_model if replay_buffer.num_stored > 10 else 1
            if env_steps % cfg.algorithm.freq_train_model == 0:
                mbrl.util.common.train_model_and_save_model_and_data(
                    train_using_only_offline_data,
                    dynamics_model,
                    model_trainer,
                    cfg.overrides,
                    replay_buffer,
                    replay_buffer_test,
                    work_dir=work_dir,
                )
                save_replay_buffer_online = True
                if env_steps != 0 and save_replay_buffer_online:
                    replay_buffer.save(work_dir, "replay_buffer_online")
            # TODO: check
            if env_steps == 0:
                train_using_only_offline_data = False
                cfg.overrides.num_epochs_train_model = default_num_epochs_train_model
                cfg.overrides.model_batch_size = 128
                if cfg.algorithm.reset_last_few_layers_of_model:
                    for param in dynamics_model.model.hidden_layers[-1].parameters():
                        param.data.normal_(mean=0, std=0.01)
                if cfg.algorithm.reset_buffer_after_offline:
                    replay_buffer.cur_idx = 0
                    replay_buffer.num_stored = 0
                if cfg.algorithm.freeze_dynamic_network_parameters and cfg.algorithm.initial_obs_steps > 10:
                    for param in dynamics_model.model.gnn_layer1.parameters():
                        param.requires_grad = False
                    for param in dynamics_model.model.gnn_layer2.parameters():
                        param.requires_grad = False
            if cfg.algorithm.reset_last_few_layers_of_model and env_steps % cfg.algorithm.reset_interval == 0:
                for param in dynamics_model.model.hidden_layers[-1].parameters():
                    param.data.normal_(mean=0, std=0.01)
            # --- Doing env step using the agent and adding to model dataset ---
            # profiler = cProfile.Profile()
            # profiler.enable()
            action, next_obs, reward, done, _ = mbrl.util.common.step_env_and_add_to_buffer(
                env, obs, agent, {}, replay_buffer, work_dir=work_dir
            )
            total_cost_reward += np.sum(np.abs(action))
            # profiler.disable()
            # profiler.print_stats(sort='cumulative')
            obs = next_obs
            total_reward += reward
            steps_trial += 1
            env_steps += 1
            print("steps_trial = ", steps_trial)
            if debug_mode:
                print(f"Step {env_steps}: Reward {reward:.3f}.")

        wandb.log({"env_step": env_steps, "episode_reward": total_reward, "total_cost_reward": total_cost_reward})
        if logger is not None:
            logger.log_data(
                mbrl.constants.RESULTS_LOG_NAME,
                {"env_step": env_steps, "episode_reward": total_reward, "total_cost_reward": total_cost_reward},
            )
        current_trial += 1
        if debug_mode:
            print(f"Trial: {current_trial }, reward: {total_reward}.")

        max_total_reward = max(max_total_reward, total_reward)

    wandb.finish()
    return np.float32(max_total_reward)
