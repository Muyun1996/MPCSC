[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://img.shields.io/badge/DOI-10.1063%2F5.0195208-blue.svg)](https://doi.org/10.1063/5.0195208)

# MBRL-CNOC: Model-Based Control of Complex Networked Systems

This is the official code repository for the paper:

> **Model predictive complex system control from observational and interventional data**<br>
> Muyun Mou, Yu Guo, Fanming Luo, Yang Yu, and Jiang Zhang<br>
> *Chaos: An Interdisciplinary Journal of Nonlinear Science* 34, 093125 (2024)<br>
> https://doi.org/10.1063/5.0195208

This repository adapts [MBRL-Lib](https://github.com/facebookresearch/mbrl-lib)
(Pineda et al., 2021) to **model-based control of complex networked systems**.
Instead of standard Mujoco locomotion tasks, it targets three canonical complex
systems defined on networks / agent populations, where only a small subset of
nodes (*driver nodes*) can be directly controlled:

| System | Environment ID | Control goal |
|---|---|---|
| **Boids** (flocking) | `gym_dynamics/Boids_dynamics` | Steer a few driver boids so that the whole flock aligns its heading (maximize the flock order parameter). |
| **Kuramoto** (coupled oscillators) | `gym_dynamics/Kuramoto_dynamics` | Apply control inputs to driver oscillators on a network to synchronize all phases (maximize the Kuramoto order parameter). |
| **SIS** (epidemic spreading) | `gym_dynamics/SIS_dynamics` | Allocate interventions (e.g. treatment/quarantine) to driver nodes on a contact network to suppress the infection (minimize the mean infected fraction). |

The control policy is **PETS** (probabilistic ensembles + CEM trajectory
optimization), where the learned dynamics model is a **graph-aware ensemble**
(`GaussianGNN`) that exploits the network structure of the system. Three model
backbones are supported via `dynamics_model.model_type`:

- `mlp` — standard fully-connected Gaussian MLP ensemble (baseline);
- `gnn` — Gaussian graph neural network ensemble;
- `sgcn` — sparse graph convolutional network (asymmetric convolution with a
  learned sparse weighted adjacency, see `mbrl/models/sgcn.py`).

The three environments are implemented as standard Gym environments in the
bundled [`gym-dynamics`](mbrl/third_party/gym-dynamics) package
(network topologies, driver-node selection and system parameters such as
coupling strength / infection rate are configurable).

## Installation

The project requires Python 3.8+ and [PyTorch](https://pytorch.org) (>= 1.7).

```bash
git clone https://github.com/Muyun1996/mbrl-cnoc.git
cd mbrl-cnoc

# install the library and its dependencies
pip install -e .
pip install -r requirements/main.txt

# install the bundled complex-system environments
pip install -e mbrl/third_party/gym-dynamics
```

For development (tests, linting):

```bash
pip install -e ".[dev]"
python -m pytest tests/core
python -m pytest tests/algorithms
```

The model-free baselines additionally require
[tianshou](https://github.com/thu-ml/tianshou) (see `baseline/ppo.py` and
`baseline/sac.py`).

## Quick start: controlling the three systems

Training is launched through the Hydra entry point `mbrl/examples/main.py`,
with one override configuration per system
(`mbrl/examples/conf/overrides/pets_{boids,kuramoto,sis}.yaml`):

```bash
# Boids flocking (default: 10 boids, 3 driver boids)
python -m mbrl.examples.main algorithm=pets overrides=pets_boids \
    dynamics_model.model_type=sgcn device=cuda:0

# Kuramoto synchronization (default: ER network, 50 nodes, 15 driver nodes)
python -m mbrl.examples.main algorithm=pets overrides=pets_kuramoto \
    dynamics_model.model_type=sgcn device=cuda:0

# SIS epidemic control (default: ER network, 20 nodes, 5 driver nodes)
python -m mbrl.examples.main algorithm=pets overrides=pets_sis \
    dynamics_model.model_type=sgcn device=cuda:0
```

Results are saved to `./exp/pets/<experiment>/<env>/<date>/<time>/` as
`results.csv`, together with model-training logs (`model_train.csv`). Use
`root_dir=...` and `experiment=...` to change the output location.

### Useful configuration options

System parameters can be changed without editing code, e.g.:

```bash
python -m mbrl.examples.main algorithm=pets overrides=pets_kuramoto \
    overrides.params_to_env.node_num=100 \
    overrides.params_to_env.driver_node_num=10 \
    overrides.params_to_env.noise_std=0.1
```

Other frequently used options (see `mbrl/examples/conf` for the full list):

- `dynamics_model.model_type={mlp,gnn,sgcn}` — dynamics model backbone;
- `dynamics_model.num_layers` — number of GNN/SGCN layers;
- `algorithm.initial_obs_steps` — random-exploration steps used to collect the
  initial training set before planning starts;
- `algorithm.epistemic_reward_weight` — weight of the epistemic-uncertainty
  bonus added to the reward during planning (exploration);
- `algorithm.use_symlog_normalizer` — symlog normalization of model inputs;
- `overrides.planning_horizon`, `overrides.cem_population_size`,
  `overrides.cem_num_iters` — CEM planner settings;
- `seed` (run seed) and `overrides.params_to_env.seed` (environment seed).

## Batch experiments

The scripts `run_boids_batch_experiments.py`, `run_kuramoto_batch_experiments.py`
and `run_sis_batch_experiments.py` at the repository root reproduce the paper
experiments. Each script contains a set of parameterized sweeps (baselines over
seeds, ablations over observation steps, epistemic reward weight, model type,
number of layers, normalization, buffer/model reset strategies, fixed driver
node number, etc.) and launches one tmux session per run, distributing runs
over the available CUDA devices. Enable the sweeps you want by flipping the
corresponding `test_*` flags at the top of each script, then run e.g.:

```bash
python run_boids_batch_experiments.py
```

## Baselines

The `baseline/` folder contains reference controllers for comparison:

- `baseline/ppo.py`, `baseline/sac.py` — model-free PPO / SAC baselines
  implemented with [tianshou](https://github.com/thu-ml/tianshou), running
  directly on the `gym_dynamics` environments;
- `baseline/data_driven_control.py` — a data-driven optimal-control baseline
  (linear-systems-based, scipy implementation) for the Kuramoto system.

## Plotting

The `draw/` folder contains the scripts used to generate the result figures
from saved experiment logs:

- `draw_baseline.py` — comparison against the baselines;
- `draw_boids_ablation.py` — ablation studies on Boids;
- `draw_barplot_different_graph_size.py` — performance vs. network size;
- `draw_barplot_noise_std.py` — performance vs. observation noise;
- `draw_comparison_of_init_obs_in_there_environment.py` — effect of the initial
  observation steps across the three systems.

## Repository structure

```
├── mbrl/                        # core library (forked from MBRL-Lib)
│   ├── algorithms/              # PETS (used here), MBPO, PlaNet
│   ├── models/                  # gaussian_gnn.py (GNN ensemble), sgcn.py, ...
│   ├── planning/                # CEM / iCEM / MPPI trajectory optimizers
│   ├── examples/                # Hydra entry point and configurations
│   └── third_party/gym-dynamics # Boids / Kuramoto / SIS Gym environments
├── baseline/                    # PPO / SAC / data-driven control baselines
├── draw/                        # plotting scripts for the result figures
├── run_*_batch_experiments.py   # batch runners for the three systems
└── tests/                       # unit tests
```

## Citation

If you find this code useful in your research, please cite our paper:

```BibTeX
@Article{Mou2024MPCSC,
  author  = {Muyun Mou and Yu Guo and Fanming Luo and Yang Yu and Jiang Zhang},
  title   = {Model predictive complex system control from observational and interventional data},
  journal = {Chaos: An Interdisciplinary Journal of Nonlinear Science},
  year    = {2024},
  volume  = {34},
  number  = {9},
  pages   = {093125},
  doi     = {10.1063/5.0195208},
  url     = {https://pubs.aip.org/aip/cha/article/34/9/093125/3313311},
}
```

## Acknowledgements

This project is built on top of
[MBRL-Lib](https://github.com/facebookresearch/mbrl-lib). The Boids simulator
is adapted from [PyNBoids](https://github.com/Nikorasu/PyNBoids). If you use
the underlying library in your research, please cite:

```BibTeX
@Article{Pineda2021MBRL,
  author  = {Luis Pineda and Brandon Amos and Amy Zhang and Nathan O. Lambert and Roberto Calandra},
  journal = {Arxiv},
  title   = {MBRL-Lib: A Modular Library for Model-based Reinforcement Learning},
  year    = {2021},
  url     = {https://arxiv.org/abs/2104.10159},
}
```

## License

This project is released under the MIT license. See [LICENSE](LICENSE) for details.
