from typing import Any, Optional
from typing_extensions import Self
from inspyred.ec.emo import Pareto
from inspyred.ec import Individual
from numpy.random import RandomState
import numpy as np
import functools
from numpy.typing import NDArray


def choice_without_replacement(rng: RandomState, n: int, size: int) -> set[int]:
    result = set()
    while len(result) < size:
        result.add(rng.randint(0, n))
    return result


class NumpyRandomWrapper(RandomState):
    def __init__(self, seed: Optional[int] = None):
        super(NumpyRandomWrapper, self).__init__(seed)

    def sample(self, pop: int | list[list[float]], k: int):
        if isinstance(pop, int):
            population = range(pop)
        else:
            population = pop

        return np.asarray(
            [
                population[i]
                for i in choice_without_replacement(self, len(population), k)
            ]
        )

    def random(self) -> float:  # type: ignore
        return self.random_sample()

    def gauss(self, mu: float, sigma: float) -> float:
        return self.normal(mu, sigma)


def initial_pop_observer(
    population: list[Individual],
    num_generations: int,
    num_evaluations: int,
    args: dict[str, Any],
):
    if num_generations == 0:
        args["initial_pop_storage"]["individuals"] = np.asarray(
            [guy.candidate for guy in population]
        )
        args["initial_pop_storage"]["fitnesses"] = np.asarray(
            [guy.fitness for guy in population]
        )


def generator(random: RandomState, args: dict[str, Any]) -> NDArray[np.float64]:
    return np.asarray(
        [
            random.uniform(args["pop_init_range"][0], args["pop_init_range"][1])
            for _ in range(args["num_vars"])
        ]
    )


def generator_wrapper(func):  # type: ignore
    @functools.wraps(func)
    def _generator(random: RandomState, args: dict[str, Any]) -> NDArray[np.float64]:
        return np.asarray(func(random, args))

    return _generator


class CombinedObjectives(Pareto):
    def __init__(self, pareto: Pareto, args: dict[str, Any]):
        """edit this function to change the way that multiple objectives
        are combined into a single objective

        """

        Pareto.__init__(self, pareto.values)
        if "fitness_weights" in args:
            weights = np.asarray(args["fitness_weights"])
        else:
            weights = np.asarray([1 for _ in pareto.values])

        self.fitness = sum(np.asarray(pareto.values) * weights)

    def __lt__(self, other: Self) -> bool:
        return self.fitness < other.fitness


def single_objective_evaluator(candidates: NDArray[np.float64], args: dict[str, Any]):
    problem = args["problem"]
    return [
        CombinedObjectives(fit, args) for fit in problem.evaluator(candidates, args)
    ]
