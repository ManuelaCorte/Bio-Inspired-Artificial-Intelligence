from functools import reduce, wraps
from typing import Any

import numpy as np
from inspyred import benchmarks
from inspyred.ec import (
    EvolutionaryComputation,
    Individual,
    replacers,
    selectors,
    terminators,
)
from numpy.random import RandomState
from numpy.typing import NDArray

from utils.inspyred_utils import (
    NumpyRandomWrapper,
    generator,
    generator_wrapper,
    initial_pop_observer,
)
from utils.plot_utils import plot_observer, plot_results_1D, plot_results_2D

GLOBAL = "Global"
INDIVIDUAL = "Individual"
CORRELATED = "Correlated"


class ES(EvolutionaryComputation):
    """Evolution Strategy EC

    This class is a stub for you to implement progressively more
    sophisticated evolution strategies.

    Optional keyword arguments in ``evolve`` args parameter that you
    will need add support for:

    - *strategy_mode* -- One of {None, 'global', 'individual', 'correlated'}
    - *epsilon* -- the minimum allowed strategy parameter (default 0.00001)
    - *tau* -- a global proportionality constant (default None)
    - *tau_i* -- an individual proportionality constant (default None)
    - *num_offspring* -- number of offspring to generate at each iteration
                        (``\\lambda``) should be a multiple of \\mu
    - *mixing_number* -- mixing number (``\\rho``) (number of parents
                        involved in producing each offspring),
                        default 1 (no-mixing)

    If *tau* is ``None``, it will be set to ``1 / sqrt(2*n)``, where
    ``n`` is the length of a candidate. If *tau_i* is ``None``, it will be
    set to ``1 / sqrt(2*sqrt(n))``.

    """

    def __init__(self, random: NumpyRandomWrapper | RandomState):
        EvolutionaryComputation.__init__(self, random)
        self.selector = selectors.default_selection
        self.variator = self._internal_variation
        # self.replacer = replacers.comma_replacement
        self.replacer = replacers.plus_replacement
        self.num_vars: int = 2

    def elementary_rotation(self, p: int, q: int, alphas: list[float]):
        R = np.ones((self.num_vars, self.num_vars))

        # taken from Schwefel et al "Contemporary Evolution Strategies"
        k = int(0.5 * (2 * self.num_vars - p - 1) * (p + 2) - 2 * self.num_vars + q)
        cos_alpha = np.cos(alphas[k])
        sin_alpha = np.sin(alphas[k])
        R[p][p] = cos_alpha
        R[q][q] = cos_alpha
        R[p][q] = -sin_alpha
        R[q][p] = sin_alpha
        return R

    def _internal_variation(
        self,
        random: NumpyRandomWrapper | RandomState,
        candidates: NDArray[np.float64],
        args: dict[str, Any],
    ) -> list[list[float]]:
        tau = args.setdefault("tau", None)
        tau_i = args.setdefault("tau", None)
        epsilon = args.setdefault("epsilon", 0.00001)

        # num_offspring (\\lambda)
        num_offspring = args.setdefault("num_offspring", len(candidates))

        mixing_number = args.setdefault("mixing_number", 1)

        if num_offspring % len(candidates) != 0:
            raise Exception(
                "num_offspring (\\lambda) should be a multiple " + "of pop_size (\\mu)"
            )

        mutants = []

        if tau is None:
            tau = 1.0 / np.sqrt(2 * self.num_vars)
        if tau_i is None:
            tau_i = 1.0 / np.sqrt(2 * np.sqrt(self.num_vars))

        while len(mutants) < num_offspring:
            parent_family_inidices = random.sample(len(candidates), mixing_number)  # type: ignore
            parent_family = np.asarray([candidates[i] for i in parent_family_inidices])
            parent = parent_family.mean(0)

            cand = parent[: self.num_vars].copy()
            if self.strategy_mode is None:
                strat = []
                sigma = args.setdefault("sigma", 1.0)
                if isinstance(random, NumpyRandomWrapper):
                    cand += random.normal(0, sigma, cand.shape)
                else:
                    for i, c in enumerate(cand):
                        cand[i] = c + random.gauss(0, sigma)  # type: ignore

            else:
                strat = parent[self.num_vars :].copy()

                if self.strategy_mode is GLOBAL:
                    sigmas = strat
                else:
                    sigmas = strat[: self.num_vars]  # view into strat

                e_global = tau * random.gauss(0, 1)  # type: ignore

                # more efficient with sumpy
                if isinstance(random, NumpyRandomWrapper):
                    sigmas *= np.exp(
                        e_global + tau_i * random.normal(0, 1, sigmas.shape)
                    )
                    sigmas = np.maximum(sigmas, epsilon)
                else:
                    for i, s in enumerate(sigmas):
                        sigmas[i] = s * np.exp(e_global + tau_i * random.gauss(0, 1))  # type: ignore
                        sigmas[i] = max(strat[i], epsilon)

                if self.strategy_mode is CORRELATED:
                    pi = np.pi
                    alphas = strat[self.num_vars :]  # another view into strat
                    beta_squared = (5.0 * np.pi / 180) ** 2  # 5 deg squared
                    if isinstance(random, NumpyRandomWrapper):
                        alphas += random.normal(0, beta_squared, alphas.shape) + pi
                        alphas %= 2 * pi
                        alphas -= pi
                    else:
                        for j, a in enumerate(alphas):
                            alphas[j] = (
                                (a + random.gauss(0, beta_squared) + pi) % (2 * pi)  # type: ignore
                            ) - pi

                if self.strategy_mode is GLOBAL:
                    sigma = sigmas[0]
                    if isinstance(random, NumpyRandomWrapper):
                        cand = cand + random.normal(0, sigma, cand.shape)
                    else:
                        for i, c in enumerate(cand):
                            cand[i] = c + random.gauss(0, sigma)  # type: ignore
                elif self.strategy_mode is INDIVIDUAL:
                    if isinstance(random, NumpyRandomWrapper):
                        cand += random.multivariate_normal(
                            np.zeros(self.num_vars), np.diag(sigmas**2)
                        )
                    else:
                        for i, (c, s) in enumerate(zip(cand, sigmas, strict=False)):
                            cand[i] = c + random.gauss(0, s)  # type: ignore
                else:
                    # build correlation matrix
                    T = reduce(
                        np.dot,
                        [
                            reduce(
                                np.dot,
                                [
                                    self.elementary_rotation(p, q, alphas)  # type: ignore
                                    for q in range(p + 1, self.num_vars)
                                ],
                            )
                            for p in range(self.num_vars - 1)
                        ],
                    )
                    if isinstance(random, NumpyRandomWrapper):
                        cand += random.multivariate_normal(
                            np.zeros(self.num_vars), np.dot(T, np.diag(sigmas**2))
                        )
                    else:
                        raise Exception(
                            "NumpyRandomWrapper required" + " for correlated mutations"
                        )

            cand = self.bounder(cand, args)  # type: ignore
            cand = np.concatenate((cand, strat))
            mutants.append(cand)

        return mutants

    def _internal_evaluator(self, func: Any) -> Any:
        @wraps(func)
        def evaluator(
            candidates: list[list[float]], args: dict[str, Any]
        ) -> list[float]:
            # convert candidates to array and then back to list
            # makes slicing easier
            return func(list(np.asarray(candidates)[:, 0 : self.num_vars]), args)

        return evaluator

    def strategize(self, generator: Any) -> Any:
        """Add strategy parameters to candidates created by a generator.

        This function decorator is used to provide a means of adding strategy
        parameters to candidates created by a generator. The generator function
        is modifed to extend the candidate with strategy parameters based on
        the strategy_mode argument passed to evolve.

        Each strategy parameter is initialized to a random value:
        in [0, 1] for ``\\sigma_i`` and in [-pi,pi] for ``\\alpha_i``

        """

        @wraps(generator)  # type: ignore
        def strategy_generator(
            random: NumpyRandomWrapper, args: dict[str, Any]
        ) -> NDArray[np.float64]:
            candidate = generator(random, args)
            if self.strategy_mode is None:
                return np.asarray(candidate)
            elif self.strategy_mode is GLOBAL:
                return np.concatenate((candidate, [random.random()]))
            else:
                sigmas = [random.random() for _ in range(self.num_vars)]
                if self.strategy_mode is INDIVIDUAL:
                    return np.concatenate((candidate, sigmas))
                elif self.strategy_mode is CORRELATED:
                    # since have python random, do it like this... would be
                    # better with numpy
                    alphas = [
                        random.uniform(-np.pi, np.pi)
                        for _ in range((self.num_vars**2 - self.num_vars) / 2)  # type: ignore
                    ]
                    return np.concatenate((candidate, alphas, sigmas))
                else:
                    raise Exception("Invalid strategy mode")

        return strategy_generator

    def evolve(
        self,
        generator: Any,
        evaluator: Any,
        pop_size: int = 100,
        seeds: int | None = None,
        maximize: bool = False,
        bounder: Any = None,
        strategy_mode: Any = None,
        num_vars: int | None = None,
        **args: Any,
    ) -> list[Individual]:
        self.strategy_mode = strategy_mode
        self.num_vars = num_vars  # type: ignore

        generator = self.strategize(generator)
        evaluator = self._internal_evaluator(evaluator)
        return EvolutionaryComputation.evolve(
            self,
            generator,
            evaluator,
            pop_size,
            maximize=maximize,
            num_vars=num_vars,
            bounder=bounder,
            **args,
        )


def run_es(
    random: NumpyRandomWrapper | RandomState,
    display: bool = False,
    num_vars: int = 0,
    problem_class: benchmarks.Benchmark = benchmarks.Sphere(),
    maximize: bool = False,
    use_bounder: bool = True,
    **kwargs: Any,
) -> tuple[NDArray[np.float64], float, list[Individual]]:
    # create dictionaries to store data about initial population, and lines
    initial_pop_storage = {}

    algorithm = ES(random)
    algorithm.terminator = terminators.generation_termination

    if display:
        algorithm.observer = [plot_observer, initial_pop_observer]  # type: ignore
    else:
        algorithm.observer = initial_pop_observer

    kwargs["num_selected"] = kwargs["pop_size"]
    problem = problem_class(num_vars)
    if use_bounder:
        if "bounder" not in kwargs:
            kwargs["bounder"] = problem.bounder
    if "pop_init_range" in kwargs:
        kwargs["generator"] = generator
    else:
        kwargs["generator"] = generator_wrapper(problem.generator)

    final_pop = algorithm.evolve(
        evaluator=problem.evaluator,
        maximize=problem.maximize,
        initial_pop_storage=initial_pop_storage,
        num_vars=num_vars,
        **kwargs,
    )

    # best_guy = final_pop[0].candidate[0:num_vars]
    # best_fitness = final_pop[0].fitness
    final_pop_fitnesses = np.asarray([guy.fitness for guy in final_pop])
    final_pop_candidates = np.asarray([guy.candidate[0:num_vars] for guy in final_pop])  # type: ignore

    sort_indexes = sorted(
        range(len(final_pop_fitnesses)), key=final_pop_fitnesses.__getitem__
    )
    final_pop_fitnesses = final_pop_fitnesses[sort_indexes]
    final_pop_candidates = final_pop_candidates[sort_indexes]

    best_guy = final_pop_candidates[0]
    best_fitness = final_pop_fitnesses[0]

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
