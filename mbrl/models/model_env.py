# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, Optional, Tuple

import gym
import numpy as np
import torch

import mbrl.types

from . import Model
import wandb

class ModelEnv:
    """Wraps a dynamics model into a gym-like environment.

    This class can wrap a dynamics model to be used as an environment. The only requirement
    to use this class is for the model to use this wrapper is to have a method called
    ``predict()``
    with signature `next_observs, rewards = model.predict(obs,actions, sample=, rng=)`

    Args:
        env (gym.Env): the original gym environment for which the model was trained.
        model (:class:`mbrl.models.Model`): the model to wrap.
        termination_fn (callable): a function that receives actions and observations, and
            returns a boolean flag indicating whether the episode should end or not.
        reward_fn (callable, optional): a function that receives actions and observations
            and returns the value of the resulting reward in the environment.
            Defaults to ``None``, in which case predicted rewards will be used.
        generator (torch.Generator, optional): a torch random number generator (must be in the
            same device as the given model). If None (default value), a new generator will be
            created using the default torch seed.
    """

    def __init__(
        self,
        env: gym.Env,
        model: Model,
        termination_fn: mbrl.types.TermFnType,
        reward_fn: Optional[mbrl.types.RewardFnType] = None,
        generator: Optional[torch.Generator] = None,
        epistemic_reward_weight: float = 0,
        action_cost_reward_weight: float = 0,
    ):
        self.dynamics_model = model
        self.termination_fn = termination_fn
        self.reward_fn = reward_fn
        self.device = model.device

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self._current_obs: torch.Tensor = None
        self._propagation_method: Optional[str] = None
        self._model_indices = None
        if generator:
            self._rng = generator
        else:
            self._rng = torch.Generator(device=self.device)
        self._return_as_np = True
        self.epistemic_reward_weight = epistemic_reward_weight
        self.action_cost_reward_weight = action_cost_reward_weight

    def reset(
        self, initial_obs_batch: np.ndarray, return_as_np: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Resets the model environment.

        Args:
            initial_obs_batch (np.ndarray): a batch of initial observations. One episode for
                each observation will be run in parallel. Shape must be ``B x D``, where
                ``B`` is batch size, and ``D`` is the observation dimension.
            return_as_np (bool): if ``True``, this method and :meth:`step` will return
                numpy arrays, otherwise it returns torch tensors in the same device as the
                model. Defaults to ``True``.

        Returns:
            (dict(str, tensor)): the model state returned by `self.dynamics_model.reset()`.
        """
        if isinstance(self.dynamics_model, mbrl.models.OneDTransitionRewardModel):
            assert len(initial_obs_batch.shape) == 2  # batch, obs_dim
        with torch.no_grad():
            model_state = self.dynamics_model.reset(
                initial_obs_batch.astype(np.float32), rng=self._rng
            )
        self._return_as_np = return_as_np
        return model_state if model_state is not None else {}

    def step(
        self,
        actions: mbrl.types.TensorType,
        model_state: Dict[str, torch.Tensor],
        sample: bool = False,
    ) -> Tuple[mbrl.types.TensorType, mbrl.types.TensorType, np.ndarray, Dict]:
        """Steps the model environment with the given batch of actions.

        Args:
            actions (torch.Tensor or np.ndarray): the actions for each "episode" to rollout.
                Shape must be ``B x A``, where ``B`` is the batch size (i.e., number of episodes),
                and ``A`` is the action dimension. Note that ``B`` must correspond to the
                batch size used when calling :meth:`reset`. If a np.ndarray is given, it's
                converted to a torch.Tensor and sent to the model device.
            model_state (dict(str, tensor)): the model state as returned by :meth:`reset()`.
            sample (bool): if ``True`` model predictions are stochastic. Defaults to ``False``.

        Returns:
            (tuple): contains the predicted next observation, reward, done flag and metadata.
            The done flag is computed using the termination_fn passed in the constructor.
        """
        assert len(actions.shape) == 2  # batch, action_dim
        with torch.no_grad():
            # if actions is tensor, code assumes it's already on self.device
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions).to(self.device)

            # obs_mean = torch.mean(model_state['obs'], dim=1, keepdim=True)
            # obs_abs = torch.abs(model_state['obs'] - obs_mean)
            # obs_abs = torch.index_select(obs_abs, dim=1, index=torch.tensor([3, 4, 5]))
            # obs_maxidx = torch.argmax(obs_abs, dim=1, keepdim=True)
            # random_loc = np.random.randint(3, size=obs_maxidx.size())
            # random_loc = torch.from_numpy(random_loc)
            # actions_mask = torch.zeros_like(actions)
            # actions_mask.scatter_(1, random_loc, 1)
            # actions = torch.mul(actions, actions_mask)

            (
                next_observs,
                pred_rewards,
                pred_terminals,
                next_model_state,
            ) = self.dynamics_model.sample(
                actions,
                model_state,
                deterministic=not sample,
                rng=self._rng,
            )
            rewards = (
                pred_rewards
                if self.reward_fn is None
                else self.reward_fn(actions, next_observs)
            )
            dones = self.termination_fn(actions, next_observs)

            if pred_terminals is not None:
                raise NotImplementedError(
                    "ModelEnv doesn't yet support simulating terminal indicators."
                )

            if self._return_as_np:
                next_observs = next_observs.cpu().numpy()
                rewards = rewards.cpu().numpy()
                dones = dones.cpu().numpy()
            return next_observs, rewards, dones, next_model_state

    def render(self, mode="human"):
        pass

    def evaluate_action_sequences(
        self,
        action_sequences: torch.Tensor,
        initial_state: np.ndarray,
        num_particles: int,
    ) -> torch.Tensor:
        """Evaluates a batch of action sequences on the model.

        Args:
            action_sequences (torch.Tensor): a batch of action sequences to evaluate.  Shape must
                be ``B x H x A``, where ``B``, ``H``, and ``A`` represent batch size, horizon,
                and action dimension, respectively.
            initial_state (np.ndarray): the initial state for the trajectories.
            num_particles (int): number of times each action sequence is replicated. The final
                value of the sequence will be the average over its particles values.

        Returns:
            (torch.Tensor): the accumulated reward for each action sequence, averaged over its
            particles.
        """
        with torch.no_grad():
            assert len(action_sequences.shape) == 3
            population_size, horizon, action_dim = action_sequences.shape
            # either 1-D state or 3-D pixel observation
            assert initial_state.ndim in (1, 3)
            tiling_shape = (num_particles * population_size,) + tuple(
                [1] * initial_state.ndim
            )
            initial_obs_batch = np.tile(initial_state, tiling_shape).astype(np.float32)
            model_state = self.reset(initial_obs_batch, return_as_np=False)
            batch_size = initial_obs_batch.shape[0]
            task_rewards = torch.zeros(batch_size, 1).to(self.device)
            total_state_elite_sumvar = torch.zeros(population_size).to(self.device)
            terminated = torch.zeros(batch_size, 1, dtype=bool).to(self.device)
            for time_step in range(horizon):
                action_for_step = action_sequences[:, time_step, :]
                action_batch = torch.repeat_interleave(
                    action_for_step, num_particles, dim=0
                )
                _, rewards, dones, model_state = self.step(
                    action_batch, model_state, sample=True
                )
                elite_mean_list = []
                elite_permutation = model_state['propagation_indices'] % self.dynamics_model.num_elites
                for elite_i in range(self.dynamics_model.num_elites):
                    elite_i_mask = (elite_permutation == elite_i)
                    elite_i_num = torch.sum(elite_i_mask.view(population_size, num_particles), dim=1, keepdim=True)
                    state_elite_i = torch.mul(_, elite_i_mask.view(-1, 1))
                    state_elite_i = state_elite_i.view(population_size, num_particles, -1)
                    state_elite_i = torch.sum(state_elite_i, dim=1, keepdim=True)
                    state_elite_i = torch.div(state_elite_i, elite_i_num.view(population_size, 1, 1))
                    elite_mean_list.append(state_elite_i)
                state_elite = torch.cat(elite_mean_list, dim=1)
                state_elite_var = torch.var(state_elite, dim=1)
                state_elite_sumvar = torch.sum(state_elite_var, dim=1)
                total_state_elite_sumvar += state_elite_sumvar
                rewards[terminated] = 0
                terminated |= dones
                task_rewards += rewards

            task_rewards = task_rewards.reshape(-1, num_particles)
            task_rewards = task_rewards.mean(dim=1)
            epistemic_rewards = self.epistemic_reward_weight * total_state_elite_sumvar
            task_rewards[task_rewards.isnan()] = 1e-10
            epistemic_rewards[epistemic_rewards.isnan()] = 1e-10

            action_cost_rewards = action_batch.reshape(-1, num_particles, action_batch.shape[-1])
            action_cost_rewards = action_cost_rewards.mean(dim=(1, 2), keepdim=False)
            action_cost_rewards = self.action_cost_reward_weight * action_cost_rewards

            total_rewards = task_rewards + epistemic_rewards + action_cost_rewards

            # print("task_rewards.mean() = ", task_rewards.mean())
            wandb.log(
                {
                    "mean_task_rewards": task_rewards.mean().item(),
                    "mean_epistemic_rewards": epistemic_rewards.mean().item(),
                    "mean_action_cost_rewards": action_cost_rewards.mean().item(),
                    "mean_total_rewards": total_rewards.mean().item(),
                }
            )

            return total_rewards
