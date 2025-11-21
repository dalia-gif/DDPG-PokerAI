# DDPG Poker AI
## Hybrid RL Agent for Texas Hold'em

This project develops an adaptive poker agent for Texas Hold'em using a hybrid approach that combines Reinforcement Learning and Supervised Learning. The core method is **Deep Deterministic Policy Gradient (DDPG)**, with pre training on human gameplay to enable richer strategies such as bluffing.

## Setup

### Project Structure

RL_Poker_Agent  
├── README.md  
├── train.py                      - Main script for running DDPG experiments  
├── config.yaml                   - Conda environment configuration  
├── datasets/  
│   ├── parsed_poker_games.pt     - Processed human gameplay (Torch tensor)  
│   └── Export_hand_histories.txt - Raw hand histories  
├── utils/  
│   ├── agents.py                 - DDPG agent and training loop  
│   ├── model.py                  - Actor and critic network definitions  
│   ├── replaybuffer.py           - Replay buffer for self play and human data  
│   └── load_human.py             - Parsing utilities for raw logs  
└── logs/  
    ├── run_baseline_no_human_data/  
    │   ├── rewards_plot.png  
    │   └── training_output.log  
    └── run_human_ratio0-8_replacable/  
        ├── rewards_plot.png  
        └── training_output.log  

### Environment Setup

```bash
# Create the Conda environment
conda env create -f config.yaml

# Activate the environment
conda activate rl
