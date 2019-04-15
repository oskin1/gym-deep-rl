import numpy as np
import gym
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import model
from memory import *
from frame_utils import stack_frames


env = gym.make('SpaceInvaders-v0')

actions = np.array(np.identity(env.action_space.n, dtype=int))

# MODEL HYPERPARAMETERS
state_size = [110, 84, 4]  # Our input is a stack of 4 frames hence 110x84x4 (Width, height, channels)
action_size = env.action_space.n
learning_rate = 0.00025

# TRAINING HYPERPARAMETERS
total_episodes = 1  # Total episodes for training
max_steps = 50000  # Max possible steps in an episode
batch_size = 64

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0  # exploration probability at start
explore_min = 0.01  # minimum exploration probability
decay_rate = 0.00001  # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.9  # Discounting rate

# MEMORY HYPERPARAMETERS
pretrain_length = batch_size  # Number of experiences stored in the Memory when initialized for the first time
memory_size = 1000000  # Number of experiences the Memory can keep

# PREPROCESSING HYPERPARAMETERS
stack_size = 4  # Number of frames stacked

# MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True

# TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
render_episode = False

target_update = 10

if __name__ == '__main__':

    state, frames_stack = stack_frames(env.reset())

    empty_state = np.zeros_like(state, dtype=np.int)

    _, in_h, in_w = state.shape

    policy_net = model.DQNetwork(state_size, action_size, in_h, in_w)
    target_net = model.DQNetwork(state_size, action_size, in_h, in_w)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())

    # Memory initialization
    memory = ReplayMemory(memory_size)

    # Initially fill the memory with random transitions
    for i in range(pretrain_length):

        random_action = random.choice(actions)
        next_state, reward, done, _ = env.step(np.argmax(random_action))

        if done:
            next_state = empty_state
            memory.push(state, random_action, next_state, reward, done)  # add experience to the memory

            # Start a new episode.
            state, frames_stack = stack_frames(env.reset())

        else:
            next_state, frames_stack = stack_frames(next_state, frames_stack)
            memory.push(state, random_action, next_state, reward, done)
            state = next_state

    steps_done = 0

    def predict_action(in_state):
        global steps_done
        eps_threshold = np.random.rand()
        exp_prob = explore_min + (explore_start - explore_min) * np.exp(-decay_rate * steps_done)

        steps_done += 1

        if exp_prob > eps_threshold:
            action_choice = random.choice(actions)
        else:
            with torch.no_grad():
                qs = policy_net(torch.from_numpy(in_state.reshape((1, *state.shape))).float())
                best_action_idx = np.argmax(qs)
                action_choice = actions[best_action_idx]

        return action_choice, exp_prob

    def optimize_model():
        transitions = memory.sample(batch_size)
        batch = Transition(*zip(*transitions))
        states_batch = torch.tensor(batch.state)
        actions_batch = torch.tensor(batch.action)
        rewards_batch = torch.tensor(batch.reward)
        next_states_batch = torch.tensor(batch.next_state)

        non_terminal_mask = ~torch.tensor(tuple(batch.done))

        current_q_values = policy_net(states_batch.float()).mul(actions_batch.float()).sum(1)

        next_q_values = rewards_batch
        next_q_values[non_terminal_mask] = target_net(next_states_batch.float()).max(1)[0]
        next_q_values_expected = (next_q_values * gamma) + rewards_batch

        loss = F.smooth_l1_loss(current_q_values, next_q_values_expected)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

    if training:
        for episode in range(total_episodes):
            episode_rewards = []
            state, frames_stack = stack_frames(env.reset())

            for step in range(max_steps):
                action, explore_prob = predict_action(state)

                next_state, reward, done, _ = env.step(np.argmax(action))
                episode_rewards.append(reward)

                if render_episode:
                    env.render()

                if done:
                    next_state = empty_state
                else:
                    next_state, frames_stack = stack_frames(next_state, frames_stack)

                memory.push(state, action, next_state, reward, done)
                state = next_state

                optimize_model()

                if done:
                    total_reward = np.sum(episode_rewards)

                    print('Episode: {}'.format(episode),
                          'Total reward: {}'.format(total_reward),
                          'Explore P: {:.4f}'.format(explore_prob))

                    break

            if episode % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

    print('Complete')
    env.render()
    env.close()
