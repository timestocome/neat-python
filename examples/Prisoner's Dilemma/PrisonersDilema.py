
# http://github.com/timestocome

# GA, like RL, are extremely sensitive to cost functions


# Prisoner's Dilemma
# https://en.wikipedia.org/wiki/Prisoner%27s_dilemma
# 2 crooks are caught, not allowed to communicate
# if both confess get 3, 3 yrs each
# if one testifies against other 0, 5 yrs
# if neither confesses or testifies 1, 1 yrs
#

# input node is other bots action last interaction if any
# number of hidden nodes can be set in config file
# output is either 'confess' or 'stay silent'


import os
import numpy as np
import random
import neat
import PrisonersDilema_visualize as visualize
import matplotlib.pyplot as plt



########################################################################################
# main code
########################################################################################

n_generations = 200         # how many times to evolve bots
bots = []                   # keep bots here to make code clearer

n_history = 3               # how far back in foe's history do we use to judge him?
                            # be sure to match this to config file num_inputs
                          
n_tries = 10                # number of random opponets to play each round


# location of config file
path = os.path.dirname(__file__)
file_name = 'configPrisonersDilema.txt'
config_file = os.path.join(path, file_name)



# confess is 0, stay silent is 1, -1 means no previous history 
C = 0
S = 1


# dictionary to store histories and other misc info as needed
history = {}

#####################################################################################
# utility functions
####################################################################################

# check my and foe's actions return -number of years I recieved 
def score(my_action, foe_action):

    if (my_action == C) and (foe_action == C):
        return -3

    if (my_action == S) and (foe_action == S):
        return  -2

    if (my_action == C) and (foe_action == S):
        return 0

    if (my_action == S) and (foe_action == C):
        return -5


# figure out best action based on number of years I recieved and return 
# it so network can be trained 
def best_action(action, score):
    
    if score > -2:    # bot picked best action
        if action == C:     return [1, 0]
        else:               return [0, 1]        
    else:               # bot chose wrong action
        if action == C:     return [0, 1]
        else:               return [1, 0]



# test against another bot 
def eval_genomes(genomes, config):

    # reset 
    n_bots = len(genomes)
    ids = []
    nets = {}

    
    # reset everything
    ids = []
    for genome_id, genome in genomes:
        
        genome.fitness = 0.
        nets[str(genome_id)] = neat.nn.FeedForwardNetwork.create(genome, config) 
        history[str(genome_id)] =  [-1] * n_history
        ids.append(genome_id)

    # randomize order in player vs foe
    random.shuffle(ids)

    

    # play a round        
    for genome_id, genome in genomes:
        for f in range(len(genomes)):
            
            foe_id, foe_genome = genomes[f]
            my_history = history[str(genome_id)]            
            foe_history = history[str(foe_id)]
       
            # run opposing player's prior actions through network and decide action to take
            my_output = nets[str(genome_id)].activate(foe_history[-n_history:])
            my_action = np.argmax(my_output)
            history[str(genome_id)].append(my_action)

            foe_output = nets[str(foe_id)].activate(my_history[-n_history:])
            foe_action = np.argmax(foe_output)
            history[str(foe_id)].append(foe_action)


            # score action
            my_score = score(my_action, foe_action)
            foe_score = score(foe_action, my_action)


            # get best action for updating genome
            my_best_action = best_action(my_action, my_score)
            foe_best_action = best_action(foe_action, foe_score)


            genome.fitness -= (np.sum(np.subtract(my_output, my_best_action)) ) **2
            foe_genome.fitness -= (np.sum(np.subtract(foe_output, foe_best_action)) ) **2
     
    


def run():
    
    # load configuration file
    config = neat.Config(neat.DefaultGenome, 
                        neat.DefaultReproduction, 
                        neat.DefaultSpeciesSet, 
                        neat.DefaultStagnation, 
                        config_file)


    # create a population
    p = neat.Population(config)
  

    # add reporter to display progress in terminal
    p.add_reporter(neat.StdOutReporter(False))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))


    # if n == None, run until solution is found
    # else run for n_generations
    winner = p.run(eval_genomes, n=n_generations)


    # disply winner
    print('\nBest genome:\n{!s}'.format(winner))


    
    # show output of winner against test data
    print('\nTest Output, Actual, Diff:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    
    test_x = [[-1, -1, -1],
            [-1, -1, 0],
            [-1, -1, 1],
            [-1, 0, 0],
            [-1, 0, 1],
            [-1, 1, 0],
            [-1, 1, 1],
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1]
            ]



    predicted = []
    for xi in test_x:
        output = winner_net.activate(xi)
        action = np.argmax(output) 
        predicted.append(action)

    print('----------    Test Winner   ---------------------------------')
    actions = ['Confess', 'Stay Silent']
    inputs = ['Unknown', 'Confess', 'Silent']

    for i in range(len(test_x)):
        history = test_x[i]
        in_0 = history[0] + 1
        in_1 = history[1] + 1
        in_2 = history[2] + 1
        print('History: %s %s %s -----> Action: %s  ' % (inputs[in_0], inputs[in_1], inputs[in_2], actions[predicted[i]]))

        



 
    node_names = {-1:'In -2', -2: 'In  -1', -3: 'In 0',  0:'Confess', 1:'Silent'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)


   
    





#######################################################################################
# run code
#######################################################################################
run()
