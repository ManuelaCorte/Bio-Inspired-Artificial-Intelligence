#!/usr/bin/env python3
#    This file is part of qdpy.
#
#    qdpy is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    qdpy is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with qdpy. If not, see <http://www.gnu.org/licenses/>.


"""A simple example to illuminate a fitness function based on a
normalised rastrigin function. The illumination process is ran
with 2 features corresponding to the first 2 values of the genomes.
It is possible to increase the difficulty of the illumination process
by using problem dimension above 3."""

from typing import Optional
from qdpy import algorithms, containers, benchmarks, plots

import numpy as np
import random
import os
import warnings

warnings.filterwarnings("ignore")


def main(
    NO_BINS: int,
    MAX_ITEMS_BIN: int,
    BUDGET: int,
    BATCH_SIZE: int,
    PROBLEM_DIM: int,
    seed: Optional[int] = None,
):
    # Find random seed
    if seed is None:
        seed = np.random.randint(1000000)
    os.makedirs("results/ex1/" + str(seed), exist_ok=True)
    # Update and print seed
    np.random.seed(seed)
    random.seed(seed)
    print("Seed: %i" % seed)

    # Create container and algorithm. Here we use MAP-Elites, by illuminating a Grid container by evolution.
    grid = containers.Grid(
        shape=(NO_BINS, NO_BINS),
        max_items_per_bin=MAX_ITEMS_BIN,
        fitness_domain=((0.0, 1.0),),
        features_domain=((0.0, 1.0), (0.0, 1.0)),
    )
    algo = algorithms.RandomSearchMutPolyBounded(
        grid,
        budget=BUDGET,
        batch_size=BATCH_SIZE,
        dimension=PROBLEM_DIM,
        optimisation_task="minimisation",
    )

    # Create a logger to pretty-print everything and generate output data files
    logger = algorithms.TQDMAlgorithmLogger(
        algo, log_base_path="results/ex1/" + str(seed)
    )

    # Define evaluation function
    eval_fn = algorithms.partial(
        benchmarks.illumination_rastrigin_normalised, nb_features=len(grid.shape)
    )

    # Run illumination process !
    algo.optimise(eval_fn)

    # Print results info
    print("\n" + algo.summary())

    # Plot the results
    plots.default_plots_grid(logger)

    print(
        "\nAll results are available in the '%s' pickle file." % logger.final_filename
    )
    print(
        f"""
    To open it, you can use the following python code:
        import pickle
        # You may want to import your own packages if the pickle file contains custom objects

        with open("{logger.final_filename}", "rb") as f:
            data = pickle.load(f)
        # ``data`` is now a dictionary containing all results, including the final container, all solutions, the algorithm parameters, etc.

        grid = data['container']
        print(grid.best)
        print(grid.best.fitness)
        print(grid.best.features)
    """
    )
