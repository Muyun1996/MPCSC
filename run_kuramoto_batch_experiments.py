import os
import time
import subprocess
import torch

test_basic = False
test_initial_obs_steps = False
test_epistemic_reward_weight = False
test_seed = False
test_model_type = False
test_use_symlog_normalizer = False
test_reset_last_few_layers_of_model = False
test_fixed_control_node_num = False
test_different_size_graph = False

is_cpu_only = False

cuda_device_count = torch.cuda.device_count()
print(f"Number of available CUDA devices: {cuda_device_count}")
use_cuda_device_count = 0 if is_cpu_only else cuda_device_count
cuda_id_list = [i for i in range(cuda_device_count)] * 50

def generate_test_basic_params_list(test_params = None, experiments_num_for_each = 5):
    params_list = []
    total_experiments_num = experiments_num_for_each
    device_list = [f"cuda:{cuda_id_list[i]}" for i in range(total_experiments_num)] if use_cuda_device_count > 0 else ["cpu"] * total_experiments_num
    env_seed = 111111
    run_seed = []
    for j in range(experiments_num_for_each):
        run_seed.append(j)

    for i in range(total_experiments_num):
        params_list.append({
            "device": device_list[i],
            "run_seed": run_seed[i],
            "env_seed": env_seed,
            "name": f"kuramoto_basic{i+1}",
        })
    return params_list

def generate_test_initial_obs_steps_params_list(test_params = None, experiments_num_for_each = 5):
    params_list = []
    if test_params is None:
        test_params = [2, 1500, 15000]
    total_experiments_num = len(test_params) * experiments_num_for_each
    device_list = [f"cuda:{cuda_id_list[i]}" for i in range(total_experiments_num)] if use_cuda_device_count > 0 else ["cpu"] * total_experiments_num
    env_seed = 111111
    model_type = "gnn"
    epistemic_reward_weight = 0
    initial_obs_steps = []
    run_seed = []
    for i in range(len(test_params)):
        for j in range(experiments_num_for_each):
            initial_obs_steps.append(test_params[i])
            run_seed.append(j)

    for i in range(total_experiments_num):
        params_list.append({
            "device": device_list[i],
            "run_seed": run_seed[i],
            "env_seed": env_seed,
            "name": f"kuramoto_obs{i+1}",
            "model_type": model_type,
            "initial_obs_steps": initial_obs_steps[i],
            "epistemic_reward_weight": epistemic_reward_weight
        })
    return params_list

def generate_test_epistemic_reward_weight_params_list(test_params = None, experiments_num_for_each = 5):
    params_list = []
    if test_params is None:
        test_params = [0, 0.05, 0.1, 0.5]
    total_experiments_num = len(test_params) * experiments_num_for_each
    device_list = [f"cuda:{cuda_id_list[i]}" for i in range(total_experiments_num)] if use_cuda_device_count > 0 else ["cpu"] * total_experiments_num
    env_seed = 111111
    model_type = "mlp"
    initial_obs_steps = 1500
    epistemic_reward_weight = []
    run_seed = []
    for i in range(len(test_params)):
        for j in range(experiments_num_for_each):
            epistemic_reward_weight.append(test_params[i])
            run_seed.append(j)

    for i in range(total_experiments_num):
        params_list.append({
            "device": device_list[i],
            "run_seed": run_seed[i],
            "env_seed": env_seed,
            "name": f"kuramoto_epistemic{i+1}",
            "model_type": model_type,
            "initial_obs_steps": initial_obs_steps,
            "epistemic_reward_weight": epistemic_reward_weight[i]
        })
    return params_list

def generate_test_seed_params_list(test_params = None, experiments_num_for_each = 1):
    params_list = []
    if test_params is None:
        test_params = [1, 11, 111, 1111, 11111, 111111]
    total_experiments_num = len(test_params) * experiments_num_for_each
    device_list = [f"cuda:{cuda_id_list[i]}" for i in range(total_experiments_num)] if use_cuda_device_count > 0 else ["cpu"] * total_experiments_num
    model_type = "mlp"
    initial_obs_steps = 1500
    epistemic_reward_weight = 0
    env_seed = []
    run_seed = []
    for i in range(len(test_params)):
        for j in range(experiments_num_for_each):
            env_seed.append(test_params[i])
            run_seed.append(j)

    for i in range(total_experiments_num):
        params_list.append({
            "device": device_list[i],
            "run_seed": run_seed[i],
            "env_seed": env_seed[i],
            "name": f"kuramoto_seed{i+1}",
            "model_type": model_type,
            "initial_obs_steps": initial_obs_steps,
            "epistemic_reward_weight": epistemic_reward_weight
        })
    return params_list

def generate_test_model_type_params_list(test_params = None, experiments_num_for_each = 3):
    params_list = []
    if test_params is None:
        test_params = ["mlp", "gnn"]
    total_experiments_num = len(test_params) * experiments_num_for_each
    device_list = [f"cuda:{cuda_id_list[i]}" for i in range(total_experiments_num)] if use_cuda_device_count > 0 else ["cpu"] * total_experiments_num
    env_seed = 111111
    initial_obs_steps = 15000
    epistemic_reward_weight = 0
    model_type = []
    run_seed = []
    for i in range(len(test_params)):
        for j in range(experiments_num_for_each):
            model_type.append(test_params[i])
            run_seed.append(j)

    for i in range(total_experiments_num):
        params_list.append({
            "device": device_list[i],
            "run_seed": run_seed[i],
            "env_seed": env_seed,
            "name": f"kuramoto_model_type{i+1}",
            "model_type": model_type[i],
            "initial_obs_steps": initial_obs_steps,
            "epistemic_reward_weight": epistemic_reward_weight
        })
    return params_list

def generate_test_use_symlog_normalizer_params_list(test_params = None, experiments_num_for_each = 3):
    params_list = []
    if test_params is None:
        test_params = [True, False]
    total_experiments_num = len(test_params) * experiments_num_for_each
    device_list = [f"cuda:{cuda_id_list[i]}" for i in range(total_experiments_num)] if use_cuda_device_count > 0 else ["cpu"] * total_experiments_num
    model_type = "gnn"
    env_seed = 111111
    initial_obs_steps = 1500
    epistemic_reward_weight = 0
    use_symlog_normalizer = []
    run_seed = []
    for i in range(len(test_params)):
        for j in range(experiments_num_for_each):
            use_symlog_normalizer.append(test_params[i])
            run_seed.append(j)

    for i in range(total_experiments_num):
        params_list.append({
            "device": device_list[i],
            "run_seed": run_seed[i],
            "env_seed": env_seed,
            "name": f"kuramoto_use_symlog_normalizer{i+1}",
            "model_type": model_type,
            "initial_obs_steps": initial_obs_steps,
            "epistemic_reward_weight": epistemic_reward_weight,
            "use_symlog_normalizer": use_symlog_normalizer[i]
        })
    return params_list

def generate_test_reset_last_few_layers_of_model_params_list(test_params = None, experiments_num_for_each = 2):
    params_list = []
    if test_params is None:
        test_params = [True, False]
    total_experiments_num = len(test_params) * experiments_num_for_each
    device_list = [f"cuda:{cuda_id_list[i]}" for i in range(total_experiments_num)] if use_cuda_device_count > 0 else ["cpu"] * total_experiments_num
    model_type = "gnn"
    env_seed = 111111
    initial_obs_steps = 1500
    epistemic_reward_weight = 0
    reset_last_few_layers_of_model = []
    run_seed = []
    for i in range(len(test_params)):
        for j in range(experiments_num_for_each):
            reset_last_few_layers_of_model.append(test_params[i])
            run_seed.append(j)

    for i in range(total_experiments_num):
        params_list.append({
            "device": device_list[i],
            "run_seed": run_seed[i],
            "env_seed": env_seed,
            "name": f"kuramoto_reset_last_few_layers_of_model{i+1}",
            "model_type": model_type,
            "initial_obs_steps": initial_obs_steps,
            "epistemic_reward_weight": epistemic_reward_weight,
            "reset_last_few_layers_of_model": reset_last_few_layers_of_model[i]
        })
    return params_list

def generate_test_fixed_control_node_num_params_list(test_params = None, experiments_num_for_each = 2):
    params_list = []
    if test_params is None:
        test_params = [1, 2, 3]
    total_experiments_num = len(test_params) * experiments_num_for_each
    device_list = [f"cuda:{cuda_id_list[i]}" for i in range(total_experiments_num)] if use_cuda_device_count > 0 else ["cpu"] * total_experiments_num
    model_type = "gnn"
    env_seed = 111111
    initial_obs_steps = 1500
    epistemic_reward_weight = 0
    is_fixed_control_node_num = True
    fixed_control_node_num = []
    run_seed = []
    for i in range(len(test_params)):
        for j in range(experiments_num_for_each):
            fixed_control_node_num.append(test_params[i])
            run_seed.append(j)

    for i in range(total_experiments_num):
        params_list.append({
            "device": device_list[i],
            "run_seed": run_seed[i],
            "env_seed": env_seed,
            "name": f"kuramoto_fixed_control_node_num{i+1}",
            "model_type": model_type,
            "initial_obs_steps": initial_obs_steps,
            "epistemic_reward_weight": epistemic_reward_weight,
            "is_fixed_control_node_num": is_fixed_control_node_num,
            "fixed_control_node_num": fixed_control_node_num[i]
        })
    return params_list


def generate_test_different_size_graph_params_list(test_params = None, experiments_num_for_each = 4):
    global test_params_node_num, test_params_driver_node_num
    params_list = []
    if test_params is None:
        test_params_node_num = [10, 20, 50]
        test_params_driver_node_num = [3, 6, 15]
    total_experiments_num = len(test_params_node_num) * experiments_num_for_each
    device_list = [f"cuda:{cuda_id_list[i]}" for i in range(total_experiments_num)] if use_cuda_device_count > 0 else ["cpu"] * total_experiments_num
    model_type = "gnn"
    env_seed = []
    initial_obs_steps = 15000
    epistemic_reward_weight = 0
    node_num = []
    driver_node_num = []
    run_seed = 1
    for i in range(len(test_params_node_num)):
        for j in range(experiments_num_for_each):
            node_num.append(test_params_node_num[i])
            driver_node_num.append(test_params_driver_node_num[i])
            env_seed.append(j)

    for i in range(total_experiments_num):
        params_list.append({
            "node_num": node_num[i],
            "driver_node_num": driver_node_num[i],
            "observe_node_num": node_num[i],
            "device": device_list[i],
            "run_seed": run_seed,
            "env_seed": env_seed[i],
            "name": f"kuramoto_different_size_graph{i+1}",
            "model_type": model_type,
            "initial_obs_steps": initial_obs_steps,
            "epistemic_reward_weight": epistemic_reward_weight,
        })
    return params_list


def create_command(params):
    device                              = 'cpu' if 'device' not in params else params['device']
    run_seed                            = 1 if 'run_seed' not in params else params['run_seed']
    is_random_init                      = True if 'is_random_init' not in params else params['is_random_init']
    net_type                            = 'ER' if 'net_type' not in params else params['net_type']
    node_num                            = 50 if 'node_num' not in params else params['node_num']
    driver_node_num                     = 15 if 'driver_node_num' not in params else params['driver_node_num']
    observe_node_num                    = 50 if 'observe_node_num' not in params else params['observe_node_num']
    avg_degree                          = 6 if 'avg_degree' not in params else params['avg_degree']
    coupling                            = 0.6 if 'coupling' not in params else params['coupling']
    noise_std                           = 0 if 'noise_std' not in params else params['noise_std']
    env_seed                            = 111111 if 'env_seed' not in params else params['env_seed']
    num_steps                           = 20000 if 'num_steps' not in params else params['num_steps']
    use_node_id                         = True if 'use_node_id' not in params else params['use_node_id']
    patience                            = 8000 if 'patience' not in params else params['patience']
    init_num_epochs_train_model         = 2000 if 'init_num_epochs_train_model' not in params else params['init_num_epochs_train_model']
    num_epochs_train_model              = 20 if 'num_epochs_train_model' not in params else params['num_epochs_train_model']
    model_batch_size                    = 128 if 'model_batch_size' not in params else params['model_batch_size']
    cem_num_iters                       = 5 if 'cem_num_iters' not in params else params['cem_num_iters']
    planning_horizon                    = 15 if 'planning_horizon' not in params else params['planning_horizon']
    cem_population_size                 = 150 if 'cem_population_size' not in params else params['cem_population_size']
    epistemic_reward_weight             = 0 if 'epistemic_reward_weight' not in params else params['epistemic_reward_weight']
    normalize                           = True if 'normalize' not in params else params['normalize']
    use_symlog_normalizer               = True if 'use_symlog_normalizer' not in params else params['use_symlog_normalizer']
    initial_obs_steps                   = 1500 if 'initial_obs_steps' not in params else params['initial_obs_steps']
    reset_buffer_after_offline          = True if 'reset_buffer_after_offline' not in params else params['reset_buffer_after_offline']
    reset_last_few_layers_of_model      = False if 'reset_last_few_layers_of_model' not in params else params['reset_last_few_layers_of_model']
    reset_interval                      = 15000 if 'reset_interval' not in params else params['reset_interval']
    freeze_dynamic_network_parameters   = False if 'freeze_dynamic_network_parameters' not in params else params['freeze_dynamic_network_parameters']
    test_steps                          = 600 if 'test_steps' not in params else params['test_steps']
    keep_last_solution                  = False if 'keep_last_solution' not in params else params['keep_last_solution']
    model_type                          = 'gnn' if 'model_type' not in params else params['model_type']
    num_layers                          = 2 if 'num_layers' not in params else params['num_layers']
    is_fixed_control_node_num           = False if 'is_fixed_control_node_num' not in params else params['is_fixed_control_node_num']
    fixed_control_node_num              = 0 if 'fixed_control_node_num' not in params else params['fixed_control_node_num']
    name                                = 'kuramoto' if 'name' not in params else params['name']

    return f"python -m mbrl.examples.main algorithm=pets \
    overrides=pets_kuramoto device='{device}' \
    seed={run_seed} \
    overrides.is_random_init={is_random_init} \
    overrides.params_to_env.net_type='{net_type}' \
    overrides.params_to_env.node_num={node_num} \
    overrides.params_to_env.driver_node_num={driver_node_num} \
    overrides.params_to_env.observe_node_num={observe_node_num} \
    overrides.params_to_env.avg_degree={avg_degree} \
    overrides.params_to_env.coupling={coupling} \
    overrides.params_to_env.noise_std={noise_std} \
    overrides.params_to_env.seed={env_seed} \
    overrides.use_node_id={use_node_id} \
    overrides.num_steps={num_steps} \
    overrides.patience={patience} \
    overrides.init_num_epochs_train_model={init_num_epochs_train_model} \
    overrides.num_epochs_train_model={num_epochs_train_model} \
    overrides.model_batch_size={model_batch_size} \
    overrides.cem_num_iters={cem_num_iters} \
    overrides.planning_horizon={planning_horizon} \
    overrides.cem_population_size={cem_population_size} \
    overrides.epistemic_reward_weight={epistemic_reward_weight} \
    algorithm.normalize={normalize} \
    algorithm.use_symlog_normalizer={use_symlog_normalizer} \
    algorithm.initial_obs_steps={initial_obs_steps} \
    algorithm.reset_buffer_after_offline={reset_buffer_after_offline} \
    algorithm.reset_last_few_layers_of_model={reset_last_few_layers_of_model} \
    algorithm.reset_interval={reset_interval} \
    algorithm.freeze_dynamic_network_parameters={freeze_dynamic_network_parameters} \
    algorithm.test_steps={test_steps} \
    algorithm.agent.keep_last_solution={keep_last_solution} \
    dynamics_model.model_type='{model_type}' \
    dynamics_model.num_layers={num_layers} \
    action_optimizer.is_fixed_control_node_num={is_fixed_control_node_num} \
    action_optimizer.fixed_control_node_num={fixed_control_node_num} \
    name='{name}'"

if __name__ == '__main__':
    params_list = []
    if test_basic:
        params0 = generate_test_basic_params_list()
        params_list.extend(params0)
    if test_initial_obs_steps:
        params1 = generate_test_initial_obs_steps_params_list()
        params_list.extend(params1)
    if test_epistemic_reward_weight:
        params2 = generate_test_epistemic_reward_weight_params_list()
        params_list.extend(params2)
    if test_seed:
        params3 = generate_test_seed_params_list()
        params_list.extend(params3)
    if test_model_type:
        params4 = generate_test_model_type_params_list()
        params_list.extend(params4)
    if test_use_symlog_normalizer:
        params5 = generate_test_use_symlog_normalizer_params_list()
        params_list.extend(params5)
    if test_reset_last_few_layers_of_model:
        params6 = generate_test_reset_last_few_layers_of_model_params_list()
        params_list.extend(params6)
    if test_fixed_control_node_num:
        params7 = generate_test_fixed_control_node_num_params_list()
        params_list.extend(params7)
    if test_different_size_graph:
        params8 = generate_test_different_size_graph_params_list()
        params_list.extend(params8)
    for param in params_list:
        print(param)

    for params in params_list:
        session_name = params["name"]
        run_cmd = create_command(params)
        cmd = f"export PATH=~/anaconda3/bin:$PATH && source ~/.bashrc && source activate && conda activate mbrl && {run_cmd}"
        subprocess.run(["tmux", "new-session", "-d", "-s", session_name])
        subprocess.run(["tmux", "send-keys", "-t", session_name, cmd, "Enter"])
        print("running ", session_name)
        time.sleep(100)

print("All tmux sessions have been created and corresponding commands are running.")
