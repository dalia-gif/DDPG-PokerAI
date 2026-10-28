# DDPG Poker AI

## Hybrid RL Agent for Texas Hold'em

This project develops an adaptive poker agent for Texas Hold'em using a hybrid approach that combines Reinforcement Learning and Supervised Learning. The core method is **Deep Deterministic Policy Gradient (DDPG)**, with pre training on human gameplay to enable richer strategies such as bluffing.

The implementation supports training with a replay buffer that can combine agent-generated transitions with preprocessed human poker data. This provides a framework for investigating how human experience can influence an agent's learning process and decision-making.

## Setup

### Project Structure

```bash
RL_Poker_Agent
├── README.md
├── train.py
├── config.yaml
├── datasets
│   ├── parsed_poker_games.pt
│   └── Export_hand_histories.txt
├── utils
│   ├── agents.py
│   ├── model.py
│   ├── replaybuffer.py
│   └── load_human.py
└── logs
    ├── run_baseline_no_human_data
    │   ├── rewards_plot.png
    │   └── training_output.log
    └── run_human_ratio0-8_replacable
        ├── rewards_plot.png
        └── training_output.log

```


### Create the Conda Environment

```bash
conda env create -f config.yaml
conda activate rl
```

### Install the Neuron Poker Environment

This project's reinforcement learning experiments are conducted within a Texas Hold'em environment compatible with the Gym interface.

The core game logic, observation space, and action space are directly based on the Neuron Poker Repository. This environment provides the necessary standardized, multi-player setting to train the DDPG agent and investigate complex strategies.

```bash
pip install git+https://github.com/dickreuter/neuron_poker.git
```

### Training

The training script supports configuration through command-line arguments, including the number of players, the number of training episodes, the learning criterion, and the use of human gameplay data.

For example, to train an agent in a two-player game:

```bash
python train.py --players 2 --num_episodes 300
```

To configure the proportion of agent-generated experience used during sampling and enable the human-data option:

```bash
python train.py --players 2 --num_episodes 300 --ratio 0.8 --replacable true
```

The script saves a reward curve and creates a timestamped output directory under logs/. It then runs the trained agent in the poker environment and records game histories in that directory
