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
            taus = 0.9 * genome[ind:ind+self.network_size] + 0.05
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

def evaluate(genome, seed = 0, graphics = False, original_reward=True, n_neurons=10):
    # create the phenotype from the genotype:
    agent = CTRNN_agent(n_neurons, genome=genome)
    # run the agent:
    if(original_reward):
        reward = run_cart.run_cart_continuous(agent, simulation_seed=seed, graphics=graphics)
    else:
        reward = run_cart.run_cart_continuous(agent, env=run_cart.CMC_original(), simulation_seed=seed, graphics=graphics)
    #print('Reward = ' + str(reward))
    return reward

def test_best(Best, original_reward=True):    
    n_tests = 30
    fit = np.zeros([n_tests,])
    for t in range(n_tests):
        fit[t] = evaluate(Best, seed = 100+t, graphics=False, original_reward=True)
    
    plt.figure()
    plt.boxplot(fit)
    plt.ylabel('Fitness')
    plt.xticks([1], ['Fitness best individual'])
    
# Parameters CTRNN:
network_size = 10
genome_size = (network_size+3)*network_size

# Evolutionary algorithm:
n_individuals = 30
n_generations = 30
p_mut = 0.05
n_best = 3

np.random.seed(6) # 0-5 do not work
original_reward = False
Population = np.random.rand(n_individuals, genome_size)
Reward = np.zeros([n_individuals,])
max_fitness = np.zeros([n_generations,])
mean_fitness = np.zeros([n_generations,])
Best = []
fitness_best = []
for g in range(n_generations):
    
    # evaluate:
    for i in range(n_individuals):
        Reward[i] = evaluate(Population[i, :], original_reward=original_reward)
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
print('Genome = ')
for gene in range(len(Best)):
    if(gene == 0):
        print('[' + str(Best[gene]) + ', ', end='');
    elif(gene == len(Best)-1):
        print(str(Best[gene]) + ']');
    else:
        print(str(Best[gene]) + ', ', end='');

plt.figure();
plt.plot(range(n_generations), mean_fitness)
plt.plot(range(n_generations), max_fitness)
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.legend(['Mean fitness', 'Max fitness'])

evaluate(Best, graphics=True)
test_best(Best)

# [0.45171336 0.56884579 0.56491725 0.52454626 0.62193989 0.93614712
# 0.11661162 0.85653936 0.8252461  0.58089882 0.49492735 0.29667685
# 0.37451896 0.28677048 0.58993811 0.40936107 0.58782825 0.80730997
# 0.01144499 0.24997571 0.96522433 0.43802989 0.85582822 0.68161953
# 0.09922401 0.00944109 0.48865537 0.67333867 0.46357003 0.13809677
# 0.61378884 0.01584258 0.40355504 0.91482147 0.104826   0.08818507
# 0.19016941 0.66526232 0.63112008 0.05886177 0.92747044 0.15039487
# 0.90581753 0.40119454 0.1405474  0.59073871 0.61410107 0.59311201
# 0.08417983 0.56294264 0.1569458  0.51982971 0.92425958 0.78642168
# 0.16405216 0.41192316 0.47124697 0.00402501 0.31529679 0.3304696
# 0.79951395 0.61514499 0.23998659 0.75974331 0.74741824 0.05070599
# 0.81349603 0.17996044 0.78620247 0.57800292 0.26622618 0.88572737
# 0.15759135 0.32968242 0.29482816 0.8173645  0.10642025 0.28714128
# 0.90271657 0.12750287 0.80017311 0.14303477 0.39791978 0.06028407
# 0.30904659 0.19234539 0.40238758 0.77492678 0.19980607 0.40104742
# 0.03819147 0.60142126 0.49926754 0.9275328  0.24308627 0.76369425
# 0.11618019 0.54042436 0.11330077 0.50906504 0.46958862 0.58632169
# 0.36078934 0.22232279 0.21377307 0.50852865 0.03993209 0.99276823
# 0.04907156 0.96038988 0.6755306  0.35899863 0.36131098 0.05907509
# 0.11244481 0.24913528 0.99368481 0.92319233 0.5096662  0.972418
# 0.29104816 0.04881569 0.92633616 0.85156212 0.62426413 0.68661007
# 0.93283555 0.68723513 0.79902284 0.49577034]