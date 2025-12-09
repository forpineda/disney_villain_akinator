# Disney Villain Akinator – Deep Q-Learning Agent
Final Project for CS 4320 – Decision Making / Reinforcement Learning
Created by Fatima Orpineda Prieto, & Citlali Mondragon

### Project Overview
This project implements a Reinforcement Learning version of the popular game Akinator, restricted to Disney villains.
The agent’s goal is to guess a secret villain by:

-Asking yes/no questions about villain attributes

-Using limited information and a cost for each question

-Making a final guess based on its learned Q-values

We designed:

-A fully custom gym-like environment (villain_env.py)

-A Deep Q-Network agent with experience replay

-Reward shaping to encourage informative questions

-Training, evaluation, and plotting scripts


### Installation Instructions
1. Clone the repository
git clone https://github.com/yourusername/disney_villain_akinator.git
cd disney_villain_akinator

2. Install dependencies [Make sure you have Python 3.9+.]   
Then run:
pip install -r requirements.txt

### How to Run the Project
1. Train the agent

This trains a DQN using reward shaping in a 20-villain environment:

#### python experiments/train_dqn.py


+ After training, a model file will appear in the project root, e.g.:

villain_dqn_shaped.pth

2. Evaluate the trained model

To watch the agent guess villains and print correct/incorrect results:

#### python experiments/eval_dqn.py


3. Plot training results

To generate a graph of returns over time:

#### python experiments/plot_training.py

4. Compare baseline vs reward shaping (optional)

#### python experiments/compare_plots.py

This produces a figure comparing two models.