from gym.envs.registration import register

register(
    id="gym_dynamics/Linear_dynamics",
    entry_point="gym_dynamics.envs:LinearEnv",
)

register(
    id="gym_dynamics/Linear_world_model_dynamics",
    entry_point="gym_dynamics.envs:LinearWMEnv",
)

register(
    id="gym_dynamics/Linear_model_based_dynamics",
    entry_point="gym_dynamics.envs:LinearMBEnv",
)

register(
    id="gym_dynamics/Spring_dynamics",
    entry_point="gym_dynamics.envs:SpringEnv",
)

register(
    id="gym_dynamics/Spring_model_based_dynamics",
    entry_point="gym_dynamics.envs:SpringMBEnv",
)

register(
    id="gym_dynamics/Spring_world_model_dynamics",
    entry_point="gym_dynamics.envs:SpringWMEnv",
)

register(
    id="gym_dynamics/Boids_dynamics",
    entry_point="gym_dynamics.envs:BoidsEnv",
)

register(
    id="gym_dynamics/Kuramoto_dynamics",
    entry_point="gym_dynamics.envs:KuramotoEnv",
    kwargs={"net_type": 'ER', "T": 150, "node_num": 10, "driver_node_num": 3, "observe_node_num": 10, "avg_degree": 6, "coupling": 2, "noise_std": 0., "seed": 1111}
)

register(
    id="gym_dynamics/SIS_dynamics",
    entry_point="gym_dynamics.envs:SISEnv",
    kwargs={"net_type": 'ER', "T": 150, "node_num": 100, "driver_node_num": 10, "observe_node_num": 100, "initial_I_node_num": 10, "beta": 0.2, "gamma": 0.4, "noise_std": 0., "seed": 1111}

)

register(
    id="gym_dynamics/Boids_dynamics",
    entry_point="gym_dynamics.envs:BoidsEnv",
    kwargs={"T": 100, "boids_num": 150, "driver_boids_num": 5, "width": 500, "height": 500, "speed": 170, "dt": 0.02, "is_wrap": True, "seed": 1111}

)