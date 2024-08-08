from dataclasses import dataclass
from random import randint
from typing import Any, Optional

from matplotlib import pyplot as plt
from numpy.typing import NDArray
import numpy as np

from utils.ga import run_ga
from utils.es import run_es
from utils.pso import run_pso
from utils.cma_es import run_cma
from utils.inspyred_utils import NumpyRandomWrapper


@dataclass(frozen=True)
class Result:
    mean_best_fitness: float
    std_best_fitness: float
    mean_best_individual: list[float]
    std_best_individual: list[float]
    all_best_fitness: list[float]

    def __repr__(self) -> str:
        return (
            f"Mean best fitness: {self.mean_best_fitness}\n"
            f"Std best fitness: {self.std_best_fitness}\n"
            f"Mean best individual: {self.mean_best_individual}\n"
            f"Std best individual: {self.std_best_individual}\n"
        )

    def __str__(self) -> str:
        return self.__repr__()


@dataclass(frozen=True)
class MultiObjectiveResult:
    best_f1_fitness: float
    best_f2_fitness: float
    best_origin_fitness: float
    best_f1_candidate: list[float]
    best_f2_candidate: list[float]
    best_origin_candidate: list[float]

    def __repr__(self) -> str:
        return (
            f"Best f1 fitness: {self.best_f1_fitness}\n"
            f"Best f2 fitness: {self.best_f2_fitness}\n"
            f"Best origin fitness: {self.best_origin_fitness}\n"
            f"Best f1 candidate: {self.best_f1_candidate}\n"
            f"Best f2 candidate: {self.best_f2_candidate}\n"
            f"Best origin candidate: {self.best_origin_candidate}\n"
        )

    def __str__(self) -> str:
        return self.__repr__()


def run_ga_simulation(
    func: Any,
    num_simulations: int,
    args: Any,
    print_plots: bool,
) -> Result:
    seeds = [randint(0, 1000) for _ in range(num_simulations)]
    best_fitnesses: list[float] = []
    best_individuals: list[NDArray[np.float64]] = []
    best_fitness: Optional[float] = None
    best_seed: Optional[int] = None

    for i in range(num_simulations):
        b_guy, b_fitness, _ = run_ga(
            NumpyRandomWrapper(seeds[i]), display=False, problem_class=func, **args
        )
        best_fitnesses.append(b_fitness)
        best_individuals.append(b_guy)
        if best_fitness is None or b_fitness < best_fitness:
            best_fitness = b_fitness
            best_seed = seeds[i]

    mean_best_fitness: float = np.mean(best_fitnesses).item()
    std_best_fitness: float = np.std(best_fitnesses).item()
    mean_best_individual: list[float] = np.mean(best_individuals, axis=0)
    std_best_individual: list[float] = np.std(best_individuals, axis=0)

    if print_plots:
        run_ga(NumpyRandomWrapper(best_seed), display=True, problem_class=func, **args)

    return Result(
        mean_best_fitness,
        std_best_fitness,
        mean_best_individual,
        std_best_individual,
        best_fitnesses,
    )


def run_es_simulation(
    func: Any,
    num_simulations: int,
    args: Any,
    print_plots: bool,
) -> Result:
    seeds = [randint(0, 1000) for _ in range(num_simulations)]
    best_fitnesses: list[float] = []
    best_individuals: list[NDArray[np.float64]] = []
    best_fitness: Optional[float] = None
    best_seed: Optional[int] = None

    for i in range(num_simulations):
        b_guy, b_fitness, _ = run_es(
            NumpyRandomWrapper(seeds[i]), display=False, problem_class=func, **args
        )
        best_fitnesses.append(b_fitness)
        best_individuals.append(b_guy)
        if best_fitness is None or b_fitness < best_fitness:
            best_fitness = b_fitness
            best_seed = seeds[i]

    mean_best_fitness: float = np.mean(best_fitnesses).item()
    std_best_fitness: float = np.std(best_fitnesses).item()
    mean_best_individual: list[float] = np.mean(best_individuals, axis=0)
    std_best_individual: list[float] = np.std(best_individuals, axis=0)

    if print_plots:
        run_es(NumpyRandomWrapper(best_seed), display=True, problem_class=func, **args)

    return Result(
        mean_best_fitness,
        std_best_fitness,
        mean_best_individual,
        std_best_individual,
        best_fitnesses,
    )


def run_cma_simulation(
    func: Any,
    num_simulations: int,
    args: Any,
    print_plots: bool,
) -> Result:
    seeds = [randint(0, 1000) for _ in range(num_simulations)]
    best_fitnesses: list[float] = []
    best_individuals: list[NDArray[np.float64]] = []
    best_fitness: Optional[float] = None
    best_seed: Optional[int] = None

    for i in range(num_simulations):
        b_guy, b_fitness = run_cma(
            NumpyRandomWrapper(seeds[i]), display=False, problem_class=func, **args
        )
        best_fitnesses.append(b_fitness)
        best_individuals.append(b_guy)
        if best_fitness is None or b_fitness < best_fitness:
            best_fitness = b_fitness
            best_seed = seeds[i]

    mean_best_fitness: float = np.mean(best_fitnesses).item()
    std_best_fitness: float = np.std(best_fitnesses).item()
    mean_best_individual: list[float] = np.mean(best_individuals, axis=0)
    std_best_individual: list[float] = np.std(best_individuals, axis=0)

    if print_plots:
        run_cma(NumpyRandomWrapper(best_seed), display=True, problem_class=func, **args)

    return Result(
        mean_best_fitness,
        std_best_fitness,
        mean_best_individual,
        std_best_individual,
        best_fitnesses,
    )


def run_pso_simulation(
    func: Any,
    num_simulations: int,
    args: Any,
    print_plots: bool,
) -> Result:
    seeds = [randint(0, 1000) for _ in range(num_simulations)]
    best_fitnesses: list[float] = []
    best_individuals: list[NDArray[np.float64]] = []
    best_fitness: Optional[float] = None
    best_seed: Optional[int] = None

    for i in range(num_simulations):
        b_guy, b_fitness, _ = run_pso(
            NumpyRandomWrapper(seeds[i]), display=False, problem_class=func, **args
        )
        best_fitnesses.append(b_fitness)
        best_individuals.append(b_guy)
        if best_fitness is None or b_fitness < best_fitness:
            best_fitness = b_fitness
            best_seed = seeds[i]

    mean_best_fitness: float = np.mean(best_fitnesses).item()
    std_best_fitness: float = np.std(best_fitnesses).item()
    mean_best_individual: list[float] = np.mean(best_individuals, axis=0)
    std_best_individual: list[float] = np.std(best_individuals, axis=0)

    if print_plots:
        run_pso(NumpyRandomWrapper(best_seed), display=True, problem_class=func, **args)

    return Result(
        mean_best_fitness,
        std_best_fitness,
        mean_best_individual,
        std_best_individual,
        best_fitnesses,
    )


def compute_pareto_front_statistics(
    final_pop_candidates: NDArray[np.float64],
    final_pop_fitnesses: NDArray[np.float64],
) -> MultiObjectiveResult:
    """Compute the Pareto front statistics

    Args:
        final_pop_candidates (NDArray[pl.float64]): final population candidates
        final_pop_fitnesses (NDArray[pl.float64]): final population fitnesses
        num_objectives (int): number of objectives

    Returns:
        Result: best f1, f2 and origin fitness and candidates
    """
    best_f1_index = np.argmin(final_pop_fitnesses.T[0])
    best_f2_index = np.argmin(final_pop_fitnesses.T[1])
    best_origin_index = np.argmin(
        np.sqrt(final_pop_fitnesses.T[0] ** 2 + final_pop_fitnesses.T[1] ** 2)
    )

    best_f1_fitness = final_pop_fitnesses[best_f1_index][0]
    best_f2_fitness = final_pop_fitnesses[best_f2_index][1]
    best_origin_fitness = np.sqrt(best_f1_fitness**2 + best_f2_fitness**2)

    best_f1_candidate = final_pop_candidates[best_f1_index]
    best_f2_candidate = final_pop_candidates[best_f2_index]
    best_origin_candidate = final_pop_candidates[best_origin_index]

    return MultiObjectiveResult(
        best_f1_fitness=best_f1_fitness,
        best_f2_fitness=best_f2_fitness,
        best_origin_fitness=best_origin_fitness,
        best_f1_candidate=best_f1_candidate,
        best_f2_candidate=best_f2_candidate,
        best_origin_candidate=best_origin_candidate,
    )


def plot_boxplot(
    data: list[list[float]], labels: list[float] | list[str] | list[int], x_axis: str
) -> None:
    fig = plt.figure("GA (best fitness)")
    ax = fig.gca()
    ax.boxplot(data)
    ax.set_xticklabels([str(label) for label in labels])
    ax.set_yscale("log")
    ax.set_xlabel(x_axis)
    ax.set_ylabel("Best fitness")
    plt.show()
