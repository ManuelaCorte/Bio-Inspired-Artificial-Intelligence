from random import Random
from typing import Any

import numpy as np
from inspyred import benchmarks
from inspyred.ec import (
    EvolutionaryComputation,
    Individual,
    replacers,
    selectors,
    terminators,
    variators,
)
from numpy.typing import NDArray

from utils.inspyred_utils import NumpyRandomWrapper
from utils.plot_utils import plot_observer, plot_results_1D, plot_results_2D


def generate_offspring(
    random: Random,
    x0: list[float] | NDArray[np.float64],  # type: ignore
    std_dev: float,
    num_offspring: int,
    display: bool,
    kwargs: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x0: NDArray[np.float64] = np.asarray(x0, dtype=np.float64)

    problem = benchmarks.Sphere(len(x0))

    parent_fitness: NDArray[np.float64] = problem.evaluator([x0], None)[0]

    algorithm = EvolutionaryComputation(random)
    algorithm.terminator = terminators.generation_termination
    algorithm.replacer = replacers.generational_replacement
    algorithm.variator = variators.gaussian_mutation  # type: ignore

    final_pop = algorithm.evolve(
        generator=(lambda random, args: x0.copy()),  # type: ignore
        evaluator=problem.evaluator,
        pop_size=num_offspring,
        maximize=False,
        max_generations=1,
        mutation_rate=1.0,
        gaussian_stdev=std_dev,
    )

    offspring_fitnesses = np.asarray([guy.fitness for guy in final_pop])
    offspring = np.asarray([guy.candidate for guy in final_pop])

    if display:
        # Plot the parent and the offspring on the fitness landscape
        # (only for 1D or 2D functions)
        if len(x0) == 1:
            plot_results_1D(
                problem,
                x0,
                parent_fitness,
                offspring,
                offspring_fitnesses,
                "Parent",
                "Offspring",
                kwargs,
            )

        elif len(x0) == 2:
            plot_results_2D(
                problem, np.asarray([x0]), offspring, "Parent", "Offspring", kwargs
            )

    return (parent_fitness, offspring_fitnesses)


def generator(random: Random, args: dict[str, Any]) -> NDArray[np.float64]:
    return np.asarray([
        random.uniform(args["pop_init_range"][0], args["pop_init_range"][1])
        for _ in range(args["num_vars"])
    ])


def initial_pop_observer(
    population: list[Individual],
    num_generations: int,
    num_evaluations: int,
    args: dict[str, Any],
) -> None:
    if num_generations == 0:
        args["initial_pop_storage"]["individuals"] = np.asarray([
            guy.candidate for guy in population
        ])
        args["initial_pop_storage"]["fitnesses"] = np.asarray([
            guy.fitness for guy in population
        ])


def run_ga(
    random: Random | NumpyRandomWrapper,
    display: bool = False,
    num_vars: int = 0,
    problem_class: Any = benchmarks.Sphere,
    maximize: bool = False,
    use_bounder: bool = True,
    **kwargs: Any,
) -> tuple[NDArray[np.float64], float, list[Individual]]:
    # create dictionaries to store data about initial population, and lines
    initial_pop_storage = {}

    algorithm = EvolutionaryComputation(random)
    algorithm.terminator = terminators.generation_termination
    algorithm.replacer = replacers.generational_replacement
    algorithm.variator = [  # type: ignore
        variators.uniform_crossover,
        variators.gaussian_mutation,
    ]
    algorithm.selector = selectors.tournament_selection

    if display:
        algorithm.observer = [plot_observer, initial_pop_observer]  # type: ignore
    else:
        algorithm.observer = initial_pop_observer

    kwargs["num_selected"] = kwargs["pop_size"]

    if "use_penalty" in kwargs:
        problem = problem_class(num_vars, use_penalty=kwargs["use_penalty"])
    else:
        problem = problem_class(num_vars)

    if use_bounder:
        if "bounder" not in kwargs:
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

    # best_guy = final_pop[0].candidate
    # best_fitness = final_pop[0].fitness
    final_pop_fitnesses: NDArray[np.float64] = np.asarray([
        guy.fitness for guy in final_pop
    ])
    final_pop_candidates: NDArray[np.float64] = np.asarray([
        guy.candidate for guy in final_pop
    ])

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
