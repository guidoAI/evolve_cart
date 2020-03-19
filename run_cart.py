# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 10:18:05 2020

User can pass an agent to this script, so that it is called every time.

@author: guido
"""

import gym

def run_cart_continuous(agent, simulation_seed=0, n_episodes=1):
    """ Runs the continous cart problem, with the agent mapping observations to actions. 
        - agent: should implement a method act(observation, reward, done).
        - simulation_seed: used to set the random seed for simulation
        - n_episodes: how many times the task is run for evaluation
    """
    
    env = gym.make('MountainCarContinuous-v0')
    env.seed(simulation_seed)

    reward = 0
    cumulative_reward = 0
    done = False

    for i in range(n_episodes):
        ob = env.reset()
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            cumulative_reward += reward
            env.render()
            if done:
                break

    env.close()    
    
    return cumulative_reward;