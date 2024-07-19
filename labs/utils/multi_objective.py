from typing import Any

import numpy as np
from inspyred.benchmarks import Benchmark
from inspyred.ec import (
    EvolutionaryComputation,
    replacers,
    selectors,
    terminators,
    variators,
)
from inspyred.ec.emo import NSGA2
from numpy.typing import NDArray

from utils.inspyred_utils import (
    NumpyRandomWrapper,
    generator,
    initial_pop_observer,
    single_objective_evaluator,
)
from utils.plot_utils import (
    plot_observer,
    plot_results_multi_objective_1D,
    plot_results_multi_objective_2D,
    plot_results_multi_objective_PF,
)


def run_nsga2(
    random: NumpyRandomWrapper,
    problem: Benchmark,
    display: bool = False,
    num_vars: int = 0,
    use_bounder: int = True,
    variator: Any = None,
    **kwargs: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run NSGA2 on the given problem"""
    # create dictionaries to store data about initial population, and lines
    initial_pop_storage = {}

    algorithm = NSGA2(random)
    algorithm.terminator = terminators.generation_termination
    if variator is None:
        algorithm.variator = [variators.blend_crossover, variators.gaussian_mutation]  # type: ignore
    else:
        algorithm.variator = variator

    kwargs["num_selected"] = kwargs["pop_size"]
    if use_bounder:
        kwargs["bounder"] = problem.bounder

    if display and problem.objectives == 2:
        algorithm.observer = [initial_pop_observer]  # type: ignore
    else:
        algorithm.observer = initial_pop_observer

    final_pop = algorithm.evolve(
        evaluator=problem.evaluator,
        maximize=problem.maximize,
        initial_pop_storage=initial_pop_storage,
        num_vars=num_vars,
        generator=problem.generator,
        **kwargs,
    )

    # best_guy = final_pop[0].candidate[0:num_vars]
    # best_fitness = final_pop[0].fitness
    # final_pop_fitnesses = asarray([guy.fitness for guy in algorithm.archive])
    # final_pop_candidates = asarray([guy.candidate[0:num_vars] for guy in algorithm.archive])
    final_pop_fitnesses = np.asarray([guy.fitness for guy in final_pop])
    final_pop_candidates = np.asarray([guy.candidate[0:num_vars] for guy in final_pop])

    if display:
        # Plot the parent and the offspring on the fitness landscape
        # (only for 1D or 2D functions)
        if num_vars == 1:
            plot_results_multi_objective_1D(
                problem,
                initial_pop_storage["individuals"],
                initial_pop_storage["fitnesses"],
                final_pop_candidates,
                final_pop_fitnesses,
                "Initial Population",
                "Final Population",
                len(final_pop_fitnesses[0]),
                kwargs,
            )

        elif num_vars == 2:
            plot_results_multi_objective_2D(
                problem,
                initial_pop_storage["individuals"],
                final_pop_candidates,
                "Initial Population",
                "Final Population",
                len(final_pop_fitnesses[0]),
                kwargs,
            )

        plot_results_multi_objective_PF(
            final_pop, kwargs["fig_title"] + " (Pareto front)"
        )

    return final_pop_candidates, final_pop_fitnesses


def run_ga(
    random: NumpyRandomWrapper,
    problem: Benchmark,
    obj: Any,
    display: bool = False,
    num_vars: int = 0,
    maximize: bool = False,
    use_bounder: bool = True,
    **kwargs: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run a GA on the given problem"""
    # create dictionaries to store data about initial population, and lines
    initial_pop_storage = {}

    algorithm = EvolutionaryComputation(random)
    algorithm.terminator = terminators.generation_termination
    algorithm.replacer = replacers.generational_replacement
    algorithm.variator = [variators.uniform_crossover, variators.gaussian_mutation]  # type: ignore
    algorithm.selector = selectors.tournament_selection
    if display and num_vars == 2:
        algorithm.observer = [plot_observer, initial_pop_observer]  # type: ignore
    else:
        algorithm.observer = [plot_observer]  # type: ignore

    kwargs["num_selected"] = kwargs["pop_size"]
    if use_bounder:
        kwargs["bounder"] = problem.bounder
    if "pop_init_range" in kwargs:
        kwargs["generator"] = generator
    else:
        kwargs["generator"] = problem.generator

    kwargs["problem"] = problem
    kwargs["class"] = obj
    final_pop = algorithm.evolve(
        evaluator=single_objective_evaluator,
        maximize=problem.maximize,
        initial_pop_storage=initial_pop_storage,
        num_vars=num_vars,
        **kwargs,
    )

    best_guy = final_pop[0].candidate
    best_fitness = final_pop[0].fitness.fitness
    final_pop_fitnesses = np.asarray([guy.fitness for guy in final_pop])
    final_pop_candidates = np.asarray([guy.candidate for guy in final_pop])

    if display:
        # Plot the parent and the offspring on the fitness landscape
        # (only for 1D or 2D functions)
        if num_vars == 1:
            plot_results_multi_objective_1D(
                problem,
                initial_pop_storage["individuals"],
                initial_pop_storage["fitnesses"],
                final_pop_candidates,
                final_pop_fitnesses,
                "Initial Population",
                "Final Population",
                len(final_pop_fitnesses[0]),
                kwargs,
            )

        elif num_vars == 2:
            plot_results_multi_objective_2D(
                problem,
                initial_pop_storage["individuals"],
                final_pop_candidates,
                "Initial Population",
                "Final Population",
                len(final_pop_fitnesses[0]),
                kwargs,
            )

    return best_guy, best_fitness
