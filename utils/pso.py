from inspyred.benchmarks import Benchmark, Sphere
from inspyred.ec import terminators, Individual
import numpy as np
from numpy.typing import NDArray
from utils.inspyred_utils import NumpyRandomWrapper, initial_pop_observer, generator
from utils.plot_utils import plot_observer, plot_results_1D, plot_results_2D
from typing import Any
from inspyred.swarm import PSO, topologies

STAR = "star"
RING = "ring"


def run_pso(
    random: NumpyRandomWrapper,
    display: bool = False,
    num_vars: int = 0,
    problem_class: Benchmark = Sphere(),
    maximize: bool = False,
    use_bounder: bool = True,
    **kwargs: Any,
) -> tuple[NDArray[np.float64], float, list[Individual]]:
    # create dictionaries to store data about initial population, and lines
    initial_pop_storage = {}

    algorithm = PSO(random)
    algorithm.topology = topologies.star_topology
    algorithm.terminator = terminators.generation_termination

    if display:
        algorithm.observer = [plot_observer, initial_pop_observer]  # type: ignore
    else:
        algorithm.observer = initial_pop_observer

    if "topology" in kwargs:
        if kwargs["topology"] is STAR:
            algorithm.topology = topologies.star_topology
        elif kwargs["topology"] is RING:
            algorithm.topology = topologies.ring_topology

    kwargs["num_selected"] = kwargs["pop_size"]
    problem: Benchmark = problem_class(num_vars)
    if use_bounder:
        kwargs["bounder"] = problem.bounder
    if "pop_init_range" in kwargs:
        kwargs["generator"] = generator
    else:
        kwargs["generator"] = problem.generator

    final_pop: list[Individual] = algorithm.evolve(
        evaluator=problem.evaluator,
        maximize=problem.maximize,
        initial_pop_storage=initial_pop_storage,
        num_vars=num_vars,
        **kwargs,
    )

    final_pop_fitnesses = np.asarray([guy.fitness for guy in final_pop])
    final_pop_candidates = np.asarray([guy.candidate for guy in final_pop])

    sort_indexes = sorted(
        range(len(final_pop_fitnesses)), key=final_pop_fitnesses.__getitem__
    )
    final_pop_fitnesses = final_pop_fitnesses[sort_indexes]
    final_pop_candidates = final_pop_candidates[sort_indexes]

    best_guy: NDArray[np.float64] = final_pop_candidates[0]
    best_fitness: float = final_pop_fitnesses[0]

    if display:
        # Plot the parent and the offspring on the fitness landscape
        # (only for 1D or 2D functions)
        if num_vars == 1:
            plot_results_1D(
                problem,
                initial_pop_storage["individuals"],
                initial_pop_storage["fitnesses"],
                final_pop_candidates,
                final_pop_fitnesses,
                "Initial Population",
                "Final Population",
                kwargs,
            )

        elif num_vars == 2:
            plot_results_2D(
                problem,
                initial_pop_storage["individuals"],
                final_pop_candidates,
                "Initial Population",
                "Final Population",
                kwargs,
            )

    return best_guy, best_fitness, final_pop
