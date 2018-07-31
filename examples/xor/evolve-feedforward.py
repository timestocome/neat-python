
# http://github.com/timestocome

#
# making a few tweaks to Neat_Python xor example
# https://github.com/CodeReclaimers/neat-python
#
#


"""
2-input XOR example -- this is most likely the simplest possible example.
"""



from __future__ import print_function
import os
import glob
import visualize



########################################################################
# move constants up here and set them as vars
########################################################################

# get local dir so can save and restore from local directory
local_dir = os.path.dirname(__file__)
checkpoint_file = local_dir + '/xor_checkpoint-'

n_generations = 300
n_save = 5               # save checkpoint every 5th generation
n_tests = 10              # how many test generations on saved network
default_fitness = 4.



#########################################################################
# data
#########################################################################
# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]



########################################################################
# code 
########################################################################
# run training data through network and evaluate 
# fitness of outputs
def eval_genomes(genomes, config):
    
    for genome_id, genome in genomes:
    
        genome.fitness = default_fitness
        net = neat.nn.FeedForwardNetwork.create(genome, config)
    
        for xi, xo in zip(xor_inputs, xor_outputs):
    
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo[0]) ** 2



def run(config_file):

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)


    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)


    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(generation_interval=n_save, filename_prefix=checkpoint_file))


    # Run for up to n_generations.
    winner = p.run(eval_genomes, n_generations)


    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))


    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))


    # -?: input nodes, ?: output node
    node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)


    # get name of last checkpoint file saved and restore from that one
    for file in glob.glob(checkpoint_file + '*'):
        last_checkpoint = file
    p = neat.Checkpointer.restore_checkpoint(last_checkpoint)
    p.run(eval_genomes, n_tests)
    

##############################################################################
# run code
##############################################################################
# Determine path to configuration file. This path manipulation is
# here so that the script will run successfully regardless of the
# current working directory.
if __name__ == '__main__':
    
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)
