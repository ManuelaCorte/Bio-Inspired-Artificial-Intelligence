from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from inspyred import benchmarks
from numpy.typing import NDArray

from utils.cma import CMAEvolutionStrategy
from utils.inspyred_utils import NumpyRandomWrapper, generator, generator_wrapper
from utils.plot_utils import plot_results_1D, plot_results_2D


def run_cma(
    random: NumpyRandomWrapper,
    display: bool = False,
    num_vars: int = 0,
    problem_class: Any = benchmarks.Sphere,
    **kwargs: Any,
) -> tuple[NDArray[np.float64], float]:
    problem = problem_class(num_vars)
    if "pop_init_range" in kwargs:
        gen = generator
    else:
        gen = generator_wrapper(problem.generator)

    es = CMAEvolutionStrategy(
        gen(random, kwargs),
        kwargs["sigma"],
        {
            "popsize": kwargs["num_offspring"],
            "seed": random.rand() * 100000,
            "CMA_mu": kwargs["pop_size"],
        },
    )
    #'CMA_elitist' : True})
    gen = 0
    initial_pop = np.zeros((kwargs["pop_size"], num_vars))
    initial_fitnesses = np.zeros(kwargs["pop_size"])
    num_evaluations = 0
    data = []
    while gen <= kwargs["max_generations"]:
        candidates = es.ask()  # get list of new solutions
        fitnesses = problem.evaluator(candidates, kwargs)

        if display:
            fitnesses_tmp = np.sort(fitnesses)
            average_fitness = np.mean(fitnesses_tmp)
            median_fitness = fitnesses[int(len(fitnesses_tmp) / 2)]
            best_fitness = fitnesses_tmp[0]
            worst_fitness = fitnesses_tmp[-1]

            num_generations = gen
            num_evaluations = es.countevals

            if num_generations == 0:
                initial_pop = np.asarray(candidates).copy()
                initial_fitnesses = np.asarray(fitnesses).copy()

                colors = ["black", "blue", "green", "red"]
                labels = ["average", "median", "best", "worst"]
                args = {}

                plt.figure(kwargs["fig_title"] + " (fitness trend)")
                plt.ion()
                data = [
                    [num_evaluations],
                    [average_fitness],
                    [median_fitness],
                    [best_fitness],
                    [worst_fitness],
                ]
                lines = []
                for i in range(4):
                    (line,) = plt.plot(
                        data[0], data[i + 1], color=colors[i], label=labels[i]
                    )
                    lines.append(line)
                args["plot_data"] = data
                args["plot_lines"] = lines
                plt.xlabel("Evaluations")
                plt.ylabel("Fitness")
            else:
                data = kwargs["plot_data"]
                data[0].append(num_evaluations)
                data[1].append(average_fitness)
                data[2].append(median_fitness)
                data[3].append(best_fitness)
                data[4].append(worst_fitness)
                lines = kwargs["plot_lines"]
                for i, line in enumerate(lines):
                    line.set_xdata(np.array(data[0]))
                    line.set_ydata(np.array(data[i + 1]))
                kwargs["plot_data"] = data
                kwargs["plot_lines"] = lines

        es.tell(candidates, fitnesses)
        gen += 1

    final_pop = np.asarray(es.ask())
    final_pop_fitnesses = np.asarray(problem.evaluator(final_pop, kwargs))

    best_guy: NDArray[np.float64] = es.best.x  # type: ignore
    best_fitness: float = es.best.f

    if display:
        ymin = min([min(d) for d in data[1:]])
        ymax = max([max(d) for d in data[1:]])
        yrange = ymax - ymin
        plt.xlim((0, num_evaluations))
        plt.ylim((ymin - 0.1 * yrange, ymax + 0.1 * yrange))
        plt.draw()
        plt.legend()

        # Plot the parent and the offspring on the fitness landscape
        # (only for 1D or 2D functions)
        if num_vars == 1:
            plot_results_1D(
                problem,
                initial_pop,  # type: ignore
                initial_fitnesses,
                final_pop,  # type: ignore
                final_pop_fitnesses,
                "Initial Population",
                "Final Population",
                {},
            )

        elif num_vars == 2:
            plot_results_2D(
                problem,
                initial_pop,  # type: ignore
                final_pop,  # type: ignore
                "Initial Population",
                "Final Population",
                {},
            )

    return best_guy, best_fitness
