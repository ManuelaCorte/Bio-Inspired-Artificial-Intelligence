import math
from typing import Any, Optional

from inspyred import ec
from inspyred.benchmarks import Benchmark

from utils.inspyred_utils import NumpyRandomWrapper


class ConstrainedBenchmark(Benchmark):
    def __init__(self, benchmark: Benchmark):
        Benchmark.__init__(self, benchmark.dimensions, benchmark.objectives)
        self.benchmark = benchmark
        self.bounder = ec.DiscreteBounder([0, 1])
        self.maximize = self.benchmark.maximize
        self.__class__.__name__ = (
            self.__class__.__name__ + " " + self.benchmark.__class__.__name__
        )
        self.constraints = []

    def constraintsEvaluator(
        self, candidates: list[list[float]], args: dict[str, Any]
    ) -> list[list[float]]:
        """The constraint evaluator function for the benchmark problem."""
        raise NotImplementedError("The constraintsEvaluator method must be overridden.")


# -----------------------------------------------------------------------
#                 SINGLE-OBJECTIVE CONSTRAINED PROBLEMS
# -----------------------------------------------------------------------

"""
    See:
    https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_constrained_optimization
    
    Note that in these examples we consider only the 2D case.
"""


class RosenbrockCubicLine(ConstrainedBenchmark):
    def __init__(
        self,
        dimensions: int = 2,
        use_penalty: bool = True,
        bounder: Optional[list[list[float]]] = None,
    ):
        Benchmark.__init__(self, dimensions)
        self.bounder = (
            ec.Bounder([-1.5, -0.5], [1.5, 2.5])
            if bounder is None
            else ec.Bounder(*bounder)
        )
        self.maximize = False
        self.global_optimum = [1 for _ in range(self.dimensions)]
        self.use_penalty = use_penalty

    def generator(
        self, random: NumpyRandomWrapper, args: dict[str, Any]
    ) -> list[float]:
        return [random.uniform(-1.5, 1.5), random.uniform(-0.5, 2.5)]

    def evaluator(
        self, candidates: list[list[float]], args: dict[str, Any]
    ) -> list[float]:
        fitness = []
        for c in candidates:
            f = self.f(c[0], c[1])
            if self.use_penalty:
                # penalty function (note that in this case we are minimizing, so we add a positive value)
                g1 = self.g1(c[0], c[1])  # <=0
                g2 = self.g2(c[0], c[1])  # <=0
                if g1 > 0 or g2 > 0:
                    f = f + max(g1, g2)
            fitness.append(f)
        return fitness

    def constraintsEvaluator(
        self, candidates: list[list[float]], args: dict[str, Any]
    ) -> list[list[float]]:
        constraints: list[list[float]] = []
        for c in candidates:
            g1 = self.g1(c[0], c[1])  # <=0
            g2 = self.g2(c[0], c[1])  # <=0
            constraints.append([g1, g2])
        return constraints

    def f(self, x: float, y: float) -> float:
        return (1 - x) ** 2 + 100 * (y - x**2) ** 2

    def g1(self, x: float, y: float) -> float:
        return (x - 1) ** 3 - y + 1

    def g2(self, x: float, y: float) -> float:
        return x + y - 2

    def printSolution(self, c: list[float]):
        f = self.f(c[0], c[1])
        g1 = self.g1(c[0], c[1])
        g2 = self.g2(c[0], c[1])
        print("f  =", f)
        print("g1 =", g1)
        print("g2 =", g2)
        if g1 > 0 or g2 > 0:
            print("(unfeasible)")
        else:
            print("(feasible)")


class RosenbrockDisk(ConstrainedBenchmark):
    def __init__(
        self,
        dimensions: int = 2,
        use_penalty: bool = True,
        bounder: Optional[list[list[float]]] = None,
    ):
        Benchmark.__init__(self, dimensions)
        self.bounder = (
            ec.Bounder([-1.5, -1.5], [1.5, 1.5])
            if bounder is None
            else ec.Bounder(*bounder)
        )
        self.maximize = False
        self.global_optimum = [1 for _ in range(self.dimensions)]
        self.use_penalty = use_penalty

    def generator(
        self, random: NumpyRandomWrapper, args: dict[str, Any]
    ) -> list[float]:
        return [random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5)]

    def evaluator(
        self, candidates: list[list[float]], args: dict[str, Any]
    ) -> list[float]:
        fitness = []
        for c in candidates:
            f = self.f(c[0], c[1])
            if self.use_penalty:
                # penalty function (note that in this case we are minimizing, so we add a positive value)
                g1 = self.g1(c[0], c[1])  # <=0
                if g1 > 0:
                    f = f + g1
            fitness.append(f)
        return fitness

    def constraintsEvaluator(
        self, candidates: list[list[float]], args: dict[str, Any]
    ) -> list[list[float]]:
        constraints = []
        for c in candidates:
            g1 = self.g1(c[0], c[1])  # <=0
            constraints.append([g1])
        return constraints

    def f(self, x: float, y: float) -> float:
        return (1 - x) ** 2 + 100 * (y - x**2) ** 2

    def g1(self, x: float, y: float) -> float:
        return x**2 + y**2 - 2

    def printSolution(self, c: list[float]):
        f = self.f(c[0], c[1])
        g1 = self.g1(c[0], c[1])
        print("f  =", f)
        print("g1 =", g1)
        if g1 > 0:
            print("(unfeasible)")
        else:
            print("(feasible)")


class MishraBirdConstrained(ConstrainedBenchmark):
    def __init__(
        self,
        dimensions: int = 2,
        use_penalty: bool = True,
        bounder: Optional[list[list[float]]] = None,
    ):
        Benchmark.__init__(self, dimensions)
        self.bounder = (
            ec.Bounder([-10.0, -6.5], [0.0, 0.0])
            if bounder is None
            else ec.Bounder(*bounder)
        )
        self.maximize = False
        self.global_optimum = [-3.1302468, -1.5821422]
        self.use_penalty = use_penalty

    def generator(
        self, random: NumpyRandomWrapper, args: dict[str, Any]
    ) -> list[float]:
        return [random.uniform(-10.0, 0.0), random.uniform(-6.5, 0.0)]

    def evaluator(
        self, candidates: list[list[float]], args: dict[str, Any]
    ) -> list[float]:
        fitness = []
        for c in candidates:
            f = self.f(c[0], c[1])
            if self.use_penalty:
                # penalty function (note that in this case we are minimizing, so we add a positive value)
                g1 = self.g1(c[0], c[1])  # <=0
                if g1 > 0:
                    f = f + g1
            fitness.append(f)
        return fitness

    def constraintsEvaluator(
        self, candidates: list[list[float]], args: dict[str, Any]
    ) -> list[list[float]]:
        constraints: list[list[float]] = []
        for c in candidates:
            g1 = self.g1(c[0], c[1])  # <=0
            constraints.append([g1])
        return constraints

    def f(self, x: float, y: float) -> float:
        alfa = (1 - math.cos(x)) ** 2
        beta = (1 - math.sin(y)) ** 2
        return (
            math.sin(y) * math.exp(alfa) + math.cos(x) * math.exp(beta) + (x - y) ** 2
        )

    def g1(self, x: float, y: float) -> float:
        return (x + 5) ** 2 + (y + 5) ** 2 - 25

    def printSolution(self, c: list[float]):
        f = self.f(c[0], c[1])
        g1 = self.g1(c[0], c[1])
        print("f  =", f)
        print("g1 =", g1)
        if g1 > 0:
            print("(unfeasible)")
        else:
            print("(feasible)")


class Townsend(ConstrainedBenchmark):
    def __init__(
        self,
        dimensions: int = 2,
        use_penalty: bool = True,
        bounder: Optional[list[list[float]]] = None,
    ):
        Benchmark.__init__(self, dimensions)
        self.bounder = (
            ec.Bounder([-2.25, -2.5], [2.5, 1.75])
            if bounder is None
            else ec.Bounder(*bounder)
        )
        self.maximize = False
        self.global_optimum = [2.0052938, 1.1944509]
        self.use_penalty = use_penalty

    def generator(
        self, random: NumpyRandomWrapper, args: dict[str, Any]
    ) -> list[float]:
        return [random.uniform(-2.25, 2.5), random.uniform(-2.5, 1.75)]

    def evaluator(
        self, candidates: list[list[float]], args: dict[str, Any]
    ) -> list[float]:
        fitness: list[float] = []
        for c in candidates:
            f = self.f(c[0], c[1])
            if self.use_penalty:
                # penalty function (note that in this case we are minimizing, so we add a positive value)
                g1 = self.g1(c[0], c[1])  # <=0
                if g1 > 0:
                    f = f + g1
            fitness.append(f)
        return fitness

    def constraintsEvaluator(
        self, candidates: list[list[float]], args: dict[str, Any]
    ) -> list[list[float]]:
        constraints: list[list[float]] = []
        for c in candidates:
            g1 = self.g1(c[0], c[1])  # <=0
            constraints.append([g1])
        return constraints

    def f(self, x: float, y: float) -> float:
        return -((math.cos((x - 0.1) * y)) ** 2) - x * math.sin(3 * x + y)

    def g1(self, x: float, y: float) -> float:
        t = math.atan2(x, y)
        return (
            x**2
            + y**2
            - (
                2 * math.cos(t)
                - 0.5 * math.cos(2 * t)
                - 0.25 * math.cos(3 * t)
                - 0.125 * math.cos(4 * t)
            )
            ** 2
            - (2 * math.sin(t)) ** 2
        )

    def printSolution(self, c: list[float]):
        f = self.f(c[0], c[1])
        g1 = self.g1(c[0], c[1])
        print("f  =", f)
        print("g1 =", g1)
        if g1 > 0:
            print("(unfeasible)")
        else:
            print("(feasible)")


class Simionescu(ConstrainedBenchmark):
    def __init__(
        self,
        dimensions: int = 2,
        use_penalty: bool = True,
        bounder: Optional[list[list[float]]] = None,
    ):
        Benchmark.__init__(self, dimensions)
        self.bounder = (
            ec.Bounder([-1.25, -1.25], [1.25, 1.25])
            if bounder is None
            else ec.Bounder(*bounder)
        )
        self.maximize = False
        self.global_optimum = (0.84852813, -0.84852813)
        # self.global_optimum = (-0.84852813,0.84852813)
        self.use_penalty = use_penalty

    def generator(
        self, random: NumpyRandomWrapper, args: dict[str, Any]
    ) -> list[float]:
        return [random.uniform(-1.25, 1.25), random.uniform(-1.25, 1.25)]

    def evaluator(
        self, candidates: list[list[float]], args: dict[str, Any]
    ) -> list[float]:
        fitness: list[float] = []
        for c in candidates:
            f = self.f(c[0], c[1])
            if self.use_penalty:
                # penalty function (note that in this case we are minimizing, so we add a positive value)
                g1 = self.g1(c[0], c[1])  # <=0
                if g1 > 0:
                    f = f + g1
            fitness.append(f)
        return fitness

    def constraintsEvaluator(
        self, candidates: list[list[float]], args: dict[str, Any]
    ) -> list[list[float]]:
        constraints: list[list[float]] = []
        for c in candidates:
            g1 = self.g1(c[0], c[1])  # <=0
            constraints.append([g1])
        return constraints

    def f(self, x: float, y: float) -> float:
        return 0.1 * x * y

    def g1(self, x: float, y: float) -> float:
        rt = 1
        rs = 0.2
        n = 8
        return x**2 + y**2 - (rt + rs * math.cos(n * math.atan(x / y))) ** 2

    def printSolution(self, c: list[float]):
        f = self.f(c[0], c[1])
        g1 = self.g1(c[0], c[1])
        print("f  =", f)
        print("g1 =", g1)
        if g1 > 0:
            print("(unfeasible)")
        else:
            print("(feasible)")


# -----------------------------------------------------------------------


class SphereCircle(ConstrainedBenchmark):
    def __init__(
        self,
        dimensions: int = 2,
        use_penalty: bool = True,
        bounder: Optional[list[list[float]]] = None,
    ):
        Benchmark.__init__(self, dimensions)
        self.bounder = (
            ec.Bounder([-5.12] * self.dimensions, [5.12] * self.dimensions)
            if bounder is None
            else ec.Bounder(*bounder)
        )
        self.maximize = True
        self.use_penalty = use_penalty

    def generator(
        self, random: NumpyRandomWrapper, args: dict[str, Any]
    ) -> list[float]:
        return [random.uniform(-5.12, 5.12) for _ in range(self.dimensions)]

    # implements the penalty function
    def evaluator(
        self, candidates: list[list[float]], args: dict[str, Any]
    ) -> list[float]:
        fitness: list[float] = []
        for c in candidates:
            f = self.f(c[0], c[1])
            if self.use_penalty:
                # penalty function (note that in this case we are maximizing, so penalty must be negative)
                g1 = self.g1(c[0], c[1])  # <=0
                if g1 > 0:
                    # try to change this penalty function to handle larger search spaces
                    f = (self.bounder.upper_bound[0] - self.bounder.lower_bound[0]) * g1  # type: ignore
            fitness.append(f)
        return fitness

    def fitnessEvaluator(
        self, candidates: list[list[float]], args: dict[str, Any]
    ) -> list[float]:
        fitness: list[float] = []
        for c in candidates:
            x = c[0]
            y = c[1]
            f = x**2 + y**2
            fitness.append(f)
        return fitness

    def constraintsEvaluator(
        self, candidates: list[list[float]], args: dict[str, Any]
    ) -> list[list[float]]:
        constraints: list[list[float]] = []
        for c in candidates:
            g1 = self.g1(c[0], c[1])  # <=0
            constraints.append([g1])
        return constraints

    def f(self, x: float, y: float) -> float:
        return x**2 + y**2

    def g1(self, x: float, y: float) -> float:
        return x**2 + y**2 - 1

    def printSolution(self, c: list[float]):
        f = self.f(c[0], c[1])
        g1 = self.g1(c[0], c[1])
        print("f  =", f)
        print("g1 =", g1)
        if g1 > 0:
            print("(unfeasible)")
        else:
            print("(feasible)")
