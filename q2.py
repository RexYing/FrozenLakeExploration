import math
import gym
from frozen_lake import *
import numpy as np
import random
import time
from utils import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def learn_Q_QLearning(env, num_episodes=10000, gamma = 0.99, lr = 0.1, e = 0.2, max_step=6):
    """Learn state-action values using the Q-learning algorithm with epsilon-greedy exploration strategy(no decay)
    Feel free to reuse your assignment1's code
    Parameters
    ----------
    env: gym.core.Environment
    	Environment to compute Q function for. Must have nS, nA, and P as attributes.
    num_episodes: int 
    	Number of episodes of training.
    gamma: float
    	Discount factor. Number in range [0, 1)
    learning_rate: float
    	Learning rate. Number in range [0, 1)
    e: float
    	Epsilon value used in the epsilon-greedy method. 
    max_step: Int
    	max number of steps in each episode
    
    Returns
    -------
    np.array
      An array of shape [env.nS x env.nA] representing state-action values
    """
    
    Q = np.zeros((env.nS, env.nA))
    ########################################################
    #                     YOUR CODE HERE                   #
    ########################################################

    decay_rate = 1
    scores = np.zeros(num_episodes, dtype=float)
    for i in range(num_episodes):
        s = env.reset()
        print('episode ' + str(i))
        done = False
        total_score = 0
        updates = []
        t = 0
        while not done and t < max_step:
            best_a = np.argmax(Q[s])
            if random.random() > e:
                a = best_a
            else:
                a = random.randint(0, env.nA-1)
            s_next, r, done, _ = env.step(a)
            updates.append((s, a, r + gamma * max(Q[s_next])))
            s = s_next
            total_score += r
            t += 1
        for s, a, q_samp in updates:
            Q[s][a] = (1 - lr) * Q[s][a] + lr * q_samp
        e = e * decay_rate
        if i == 0:
            scores[0] = total_score
        else:
            scores[i] = scores[i-1] + total_score
    scores /= np.arange(1, num_episodes + 1)

    plt.plot(scores)
    plt.ylabel('average score')
    plt.xlabel('episodes')
    plt.savefig('q_learning_avg_scores_lr=3e-1_e=8e-1.png')

    ########################################################
    #                     END YOUR CODE                    #
    ########################################################
    return Q



def main():
	env = FrozenLakeEnv(is_slippery=False)
	Q = learn_Q_QLearning(env, num_episodes = 10000, gamma = 0.99, lr = 0.3, e = 0.85)
	render_single_Q(env, Q)


if __name__ == '__main__':
	main()
