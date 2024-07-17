"""
2-input XOR
"""

from __future__ import print_function
import os
from typing import Any
import neat
from utils.plot_utils import draw_net, plot_stats, plot_species
import matplotlib.pyplot as plt
from neat import Config


# 2-input XOR inputs and expected outputs.
inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]

num_generations = 100
num_runs = 1

config_files = [
    "datatsets/lab08/config-feedforward-2input-xor-noelitism.txt",
    "datatsets/lab08/config-feedforward-2input-xor-elitism.txt",
]


def eval_genomes(genomes: Any, config: Config) -> None:
    for _, genome in genomes:
        genome.fitness = len(inputs)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(inputs, outputs):
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo[0]) ** 2


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)

    if num_runs == 1:
        # Load configuration.
        config_file = os.path.join(local_dir, config_files[0])
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_file,
        )

        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)

        # Add a stdout reporter to show progress in the terminal.
        stats = neat.StatisticsReporter()
        p.add_reporter(neat.StdOutReporter(True))
        p.add_reporter(stats)

        # run NEAT for num_generations
        winner = p.run(eval_genomes, num_generations)

        # Display the winning genome.
        print("\nBest genome:\n{!s}".format(winner))

        # Show output of the most fit genome against training data.
        print("\nOutput:")
        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        for xi, xo in zip(inputs, outputs):
            output = winner_net.activate(xi)
            print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

        node_names = {-1: "A", -2: "B", 0: "A XOR B"}
        draw_net(
            config, winner, filename="2-input OR", view=True, node_names=node_names
        )
        plot_stats(stats, ylog=False, view=True)
        plot_species(stats, view=True)

    else:
        results = []
        for file in config_files:
            # Load configuration.
            config_file = os.path.join(local_dir, file)
            config = neat.Config(
                neat.DefaultGenome,
                neat.DefaultReproduction,
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation,
                config_file,
            )

            best_fitnesses: list[float] = []
            for i in range(num_runs):
                print("{0}/{1}".format(i + 1, num_runs))
                p = neat.Population(config)
                winner = p.run(eval_genomes, num_generations)
                best_fitnesses.append(winner.fitness)  # type: ignore
            results.append(best_fitnesses)

        fig = plt.figure("NEAT")
        ax = fig.gca()
        ax.boxplot(results)
        ax.set_xticklabels(["Without elitism", "With elitism"])
        # ax.set_yscale('log')
        ax.set_xlabel("Condition")
        ax.set_ylabel("Best fitness")
        plt.show()
