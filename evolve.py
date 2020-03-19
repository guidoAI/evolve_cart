# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 22:04:38 2020

Evolve CTRNNs for the mountain car task

@author: guido
"""

from matplotlib import pyplot as plt
from CTRNN import CTRNN
from scipy.sparse import csr_matrix
import run_cart
import gym
import numpy as np

# added unpacking of genome:
class CTRNN_agent(object):
    
    """ Continuous Time Recurrent Neural Network agent. """
    
    n_observations = 2;
    n_actions = 1;
    
    def __init__(self, network_size, genome = [], weights=[], taus = [], gains = [], biases = []):
        
        self.network_size = network_size;
        if(self.network_size < self.n_observations + self.n_actions):
            self.network_size = self.n_observations + self.n_actions;
        self.cns = CTRNN(self.network_size, step_size=0.1) 
        
        if(len(genome) == self.network_size*self.network_size+3*self.network_size):
            # Get the network parameters from the genome:
            weight_range = 3
            ind = self.network_size*self.network_size
            w = weight_range * (2.0 * (genome[:ind] - 0.5))
            weights = np.reshape(w, [self.network_size, self.network_size])
            biases = weight_range * (2.0 * (genome[ind:ind+self.network_size] - 0.5))
            ind += self.network_size
            taus = genome[ind:ind+self.network_size] + 1e-5
            ind += self.network_size
            gains = 2.0 * (genome[ind:ind+self.network_size]-0.5)
        
        if(len(weights) > 0):
            # weights must be a matrix size: network_size x network_size
            self.cns.weights = csr_matrix(weights)
        if(len(biases) > 0):
            self.cns.biases = biases
        if(len(taus) > 0):
            self.cns.taus = taus
        if(len(gains) > 0):
            self.gains = gains
    
    def act(self, observation, reward, done):
        external_inputs = np.asarray([0.0]*self.network_size)
        external_inputs[0:self.n_observations] = observation
        self.cns.euler_step(external_inputs)
        output = 2.0 * (self.cns.outputs[-self.n_actions:] - 0.5)
        return output

def evaluate(genome, seed = 0):
    # create the phenotype from the genotype:
    agent = CTRNN_agent(n_neurons, genome=genome)
    # run the agent:
    reward = run_cart.run_cart_continuous(agent, simulation_seed=seed, graphics=False)
    #print('Reward = ' + str(reward))
    return reward
    
    
# Parameters CTRNN:
network_size = 10
genome_size = (network_size+3)*network_size

# Evolutionary algorithm:
n_individuals = 30
n_generations = 30
p_mut = 0.05
n_best = 3

Population = np.random.rand(n_individuals, genome_size)
Reward = np.zeros([n_individuals,])
max_fitness = np.zeros([n_generations,])
mean_fitness = np.zeros([n_generations,])
Best = []
fitness_best = []
for g in range(n_generations):
    
    # evaluate:
    for i in range(n_individuals):
        Reward[i] = evaluate(Population[i, :])
    mean_fitness[g] = np.mean(Reward)
    max_fitness[g] = np.max(Reward)
    print('Generation {}, mean = {} max = {}'.format(g, mean_fitness[g], max_fitness[g]))
    # select:
    inds = np.argsort(Reward)
    inds = inds[-n_best:]
    if(len(Best) == 0 or Reward[-1] > fitness_best):
        Best = Population[inds[-1], :]
        fitness_best = Reward[-1]
    # vary:
    NewPopulation = np.zeros([n_individuals, genome_size])
    for i in range(n_individuals):
        ind = inds[i % n_best]
        NewPopulation[i,:] = Population[ind, :]
        for gene in range(genome_size):
            if(np.random.rand() <= p_mut):
                NewPopulation[i,gene] = np.random.rand()
    Population = NewPopulation

print('Best fitness ' + str(fitness_best))
print('Genome = ' + str(Best))

plt.figure();
plt.plot(range(n_generations), mean_fitness)
plt.plot(range(n_generations), max_fitness)
