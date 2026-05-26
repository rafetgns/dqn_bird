import gymnasium as gym
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import random
import torch
from torch import nn
import yaml

from src.experience_replay import ReplayMemory
from src.dqn import DQN

from datetime import datetime, timedelta
import argparse
import itertools

import flappy_bird_gymnasium
import os


DATE_FORMAT = "%m-%d %H:%M:%S"

# saving run informations
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

matplotlib.use('Agg')
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'hyperparameters.yml')

device = 'cpu' # GPU overhead not worth for this env


class Agent():
    
    def __init__(self, hyperparameter_set):
        with open(CONFIG_PATH, 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set

        # hyperparameters 
        self.env_id             = hyperparameters['env_id']
        self.learning_rate_a    = hyperparameters['learning_rate_a']        # learning rate (alpha)
        self.discount_factor_g  = hyperparameters['discount_factor_g']      # discount rate (gamma)
        self.network_sync_rate  = hyperparameters['network_sync_rate']      # number of steps before syncing the policy and target network
        self.replay_memory_size = hyperparameters['replay_memory_size']     # size of replay memory
        self.mini_batch_size    = hyperparameters['mini_batch_size']        # size of the batch sampled from the replay memory
        self.epsilon_init       = hyperparameters['epsilon_init']           # 100% random actions
        self.epsilon_decay      = hyperparameters['epsilon_decay']          # epsilon decay rate
        self.epsilon_min        = hyperparameters['epsilon_min']            # minimum epsilon value
        self.stop_on_reward     = hyperparameters['stop_on_reward']         # stop training after reaching this number of rewards
        self.fc1_nodes          = hyperparameters['fc1_nodes']
        self.fc2_nodes          = hyperparameters['fc2_nodes']
        self.num_episodes       = hyperparameters['num_episodes']            
        self.env_make_params    = hyperparameters.get('env_make_params', {}) # optional env-specific parameters, default to empty dict

        # neural net
        self.loss_fn   = nn.SmoothL1Loss() # huber loss
        self.optimizer = None              # optimizer can be initialized later

        # path to run info
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.INIT_MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}_init.pt')
        self.MID_TRAINING_MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}_mid.pt')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

    def run(self, is_training=True, render=False):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time
            print(f"{start_time.strftime(DATE_FORMAT)}: Training starting...")

        # create instance of the environment.
        # unpack env-specific parameters from hyperparameters.yml.
        env = gym.make(self.env_id, render_mode='human' if render else None, **self.env_make_params)

        # number of possible actions
        num_actions = env.action_space.n

        # size of the observation space
        num_states = env.observation_space.shape[0]

        # list to keep track of episode rewards
        rewards_per_episode = []

        
        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.fc2_nodes).to(device)
            

        if is_training:
            print(f"{datetime.now().strftime(DATE_FORMAT)}: Saving the initial model")
            torch.save(policy_dqn.state_dict(), self.INIT_MODEL_FILE)
            
            epsilon = self.epsilon_init
            memory = ReplayMemory(self.replay_memory_size)

            target_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.fc2_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

            # list to keep track of epsilon decay
            epsilon_history = []

            # track number of steps taken (used for syncing networks)
            step_count = 0

            best_avg_reward = -9999999
        else:
            # load learned policy to the policy network
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, weights_only=True))
            policy_dqn.eval()

        #train for n episodes
        
        for episode in range(self.num_episodes):
            state, _ = env.reset()  # initialize environment, get the initial state s_0
            state    = torch.tensor(state, dtype=torch.float, device=device)

            terminated     = False   # true when agent reaches goal or fails
            episode_reward = 0.0     # accumulate rewards for the current episode

            # interact with env until episode terminates or reaches max rewards
            # stop on reward is necessary because on some envs, it is possible for the agent to train to a point where it never terminates
            while not terminated and episode_reward < self.stop_on_reward:

                # epsilon-greedy action selection
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    # best action with policy network
                    with torch.no_grad():
                        # state.unsqueeze(dim=0): Pytorch expects a batch layer, so add batch dimension i.e. tensor([1, 2, 3]) unsqueezes to tensor([[1, 2, 3]])
                        # policy_dqn returns tensor([[1], [2], [3]]), so squeeze it to tensor([1, 2, 3])
                        # argmax finds the index of the largest element / best action
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # execute action
                new_state, reward, terminated, truncated, info = env.step(action.item())

                episode_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward    = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    memory.append((state, action, new_state, reward, terminated))
                    step_count += 1

                state = new_state

            rewards_per_episode.append(episode_reward)

            # save model when new best reward is obtained
            if is_training:
                #if the half of the episodes are played, save the current policy for comparison at the end
                if episode == self.num_episodes // 2:
                    print(f"{datetime.now().strftime(DATE_FORMAT)}: Mid training reached, best reward {episode_reward:0.1f}")
                    torch.save(policy_dqn.state_dict(), self.MID_TRAINING_MODEL_FILE)
                
                # Only calculate average after we have 50 episodes of history
                if len(rewards_per_episode) >= 50:
                    # Calculate the mean of the last 50 episodes
                    avg_reward = np.mean(rewards_per_episode[-50:])
                    
                    if avg_reward > best_avg_reward:
                        print(f"{datetime.now().strftime(DATE_FORMAT)}: New best AVERAGE reward {avg_reward:0.1f} at episode {episode}. Saving model...")
                        torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                        best_avg_reward = avg_reward

                # update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                # if enough experience is collected, optimize the policy network
                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

              

                    # sync networks after certain step 
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

                
            if is_training and len(memory) > self.mini_batch_size:
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                epsilon_history.append(epsilon)
        env.close()

    def save_graph(self, rewards_per_episode, epsilon_history):
        fig = plt.figure(1)

        # plot average rewards (y-axis) vs episodes (x-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # plot epsilon decay (y-axis) vs episodes (x-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

    # optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        self.optimizer.zero_grad()  # clear gradients
        
        # unzip the mini batch and separate the features
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # stack tensors to create batch tensors
        states       = torch.stack(states)
        actions      = torch.stack(actions)
        new_states   = torch.stack(new_states)
        rewards      = torch.stack(rewards)
        terminations = torch.tensor(terminations, dtype=torch.float, device=device)

        with torch.no_grad():
            # calculate target Q values (expected returns) with forward pass
            target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]
            '''
                target_dqn(new_states) outputs tensor of shape (batch, action) representing prediction action-values 
                    .max(dim=1)        finds the maximum q values and returs (values, indices)
                        [0]            gets the values
                
                (1-terminations)       boolean mask which zeroes out the expected future reward if transition resulted in a terminal state
            '''

        # calculate Q values from current policy with forward pass
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        # calculate loss
        loss = self.loss_fn(current_q, target_q)

        # backward pass
        loss.backward()             
        torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), max_norm=10)  # against exploding gradients
        
        # update the parameters of the policy network 
        self.optimizer.step()       

if __name__ == '__main__':
    # parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    
    # use nargs='?' so the positional argument becomes optional and respects the default value
    # which is the flappy bird configuration
    parser.add_argument(
        'hyperparameters', 
        help='The key of the hyperparameter set in hyperparameters.yml', 
        nargs='?', 
        default='flappybird1'
        )

    # switch between training and testing the model
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)