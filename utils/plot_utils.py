from __future__ import print_function
from typing import Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import inspyred.ec.analysis
from inspyred.ec import Individual
from inspyred.benchmarks import Benchmark
from neat import StatisticsReporter, Config
from numpy.typing import NDArray
import numpy
from utils.inspyred_utils import single_objective_evaluator
import copy
import graphviz
from deap.tools import Logbook
import pygraphviz as pgv
import networkx as nx


def plot_1D(axis: Axes, problem: Benchmark, x_limits: list[float]) -> None:
    dx = (x_limits[1] - x_limits[0]) / 200.0
    x = np.arange(x_limits[0], x_limits[1] + dx, dx)
    x = x.reshape(len(x), 1)
    y = problem.evaluator(x, None)
    axis.plot(x, y, "-b")


def plot_2D(axis: Axes, problem: Benchmark, x_limits: list[float]) -> Any:
    dx = (x_limits[1] - x_limits[0]) / 50.0
    x = np.arange(x_limits[0], x_limits[1] + dx, dx)
    z = np.asarray([problem.evaluator([[i, j] for i in x], None) for j in x])
    return axis.contourf(x, x, z, 64, cmap=plt.cm.hot_r)  # type: ignore


def plot_results_1D(
    problem: Benchmark,
    individuals_1: NDArray[np.float64],
    fitnesses_1: NDArray[np.float64],
    individuals_2: NDArray[np.float64],
    fitnesses_2: NDArray[np.float64],
    title_1: str,
    title_2: str,
    args: dict[str, Any],
):
    fig = plt.figure(args["fig_title"] + " (initial and final population)")
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(individuals_1, fitnesses_1, ".b", markersize=7)
    lim = max(np.array(list(map(abs, ax1.get_xlim()))))

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(individuals_2, fitnesses_2, ".b", markersize=7)
    lim = max([lim] + np.array(list(map(abs, ax2.get_xlim()))))

    ax1.set_xlim(-lim, lim)
    ax2.set_xlim(-lim, lim)

    plot_1D(ax1, problem, [-lim, lim])
    plot_1D(ax2, problem, [-lim, lim])

    ax1.set_ylabel("Fitness")
    ax2.set_ylabel("Fitness")
    ax1.set_title(title_1)
    ax2.set_title(title_2)
    fig.tight_layout()


def plot_results_2D(
    problem: Benchmark,
    individuals_1: NDArray[np.float64],
    individuals_2: NDArray[np.float64],
    title_1: str,
    title_2: str,
    args: dict[str, Any],
):
    fig = plt.figure(args["fig_title"] + " (initial and final population)")
    ax1 = fig.add_subplot(2, 1, 1, aspect="equal")
    ax1.plot(individuals_1[:, 0], individuals_1[:, 1], ".b", markersize=7)
    lim = max(
        np.array(list(map(abs, ax1.get_xlim())))
        + np.array(list(map(abs, ax1.get_ylim())))
    )

    ax2 = fig.add_subplot(2, 1, 2, aspect="equal")
    ax2.plot(individuals_2[:, 0], individuals_2[:, 1], ".b", markersize=7)
    lim = max(
        [lim]
        + np.array(list(map(abs, ax2.get_xlim())))
        + np.array(list(map(abs, ax2.get_ylim())))
    )

    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.set_title(title_1)
    ax1.locator_params(nbins=5)

    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    ax2.set_title(title_2)
    ax2.set_xlabel("x0")
    ax2.set_ylabel("x1")
    ax2.locator_params(nbins=5)

    plot_2D(ax1, problem, [-lim, lim])
    c = plot_2D(ax2, problem, [-lim, lim])
    fig.subplots_adjust(right=0.8)
    fig.tight_layout()
    cax = fig.add_axes((0.85, 0.15, 0.05, 0.7))
    colorbar_ = plt.colorbar(c, cax=cax)
    colorbar_.ax.set_ylabel("Fitness")


"""
    multi-objective plotting utils
"""


def plot_multi_objective_1D(
    axis: Axes, problem: Benchmark, x_limits: list[float], objective: float
):
    dx = (x_limits[1] - x_limits[0]) / 200.0
    x = np.arange(x_limits[0], x_limits[1] + dx, dx)
    x = x.reshape(len(x), 1)
    y = [f[objective] for f in problem.evaluator(x, None)]
    axis.plot(x, y, "-b")


def plot_multi_objective_2D(
    axis: Axes, problem: Benchmark, x_limits: list[float], objective: float
):
    dx = (x_limits[1] - x_limits[0]) / 50.0
    x = np.arange(x_limits[0], x_limits[1] + dx, dx)
    z = np.asarray([problem.evaluator([[i, j] for i in x], None) for j in x])[
        :, :, int(objective)
    ]

    return axis.contourf(x, x, z, 64, cmap=plt.cm.hot_r)  # type: ignore


def plot_results_multi_objective_1D(
    problem: Benchmark,
    individuals_1: NDArray[np.float64],
    fitnesses_1: NDArray[np.float64],
    individuals_2: NDArray[np.float64],
    fitnesses_2: NDArray[np.float64],
    title_1: str,
    title_2: str,
    num_objectives: int,
    args: dict[str, Any],
):
    fig = plt.figure(args["fig_title"] + " (initial and final population)")
    lim = None
    axes = []
    for objective in range(num_objectives):
        ax1 = fig.add_subplot(num_objectives, 2, 2 * objective + 1)
        ax1.plot(individuals_1, [f[objective] for f in fitnesses_1], ".b", markersize=7)
        if lim is None:
            lim = max(list(map(abs, ax1.get_xlim())))
        else:
            lim = max([lim] + list(map(abs, ax1.get_xlim())))

        ax2 = fig.add_subplot(num_objectives, 2, 2 * objective + 2)
        ax2.plot(individuals_2, [f[objective] for f in fitnesses_2], ".b", markersize=7)
        lim = max([lim] + list(map(abs, ax2.get_xlim())))
        axes.append(ax1)
        axes.append(ax2)
        ax1.set_title(title_1)
        ax2.set_title(title_2)
        ax1.set_ylabel("Objective " + str(objective + 1))
        ax2.set_ylabel("Objective " + str(objective + 1))

    for i, ax in enumerate(axes):
        ax.set_xlim(-lim, lim)  # type: ignore
        plot_multi_objective_1D(ax, problem, [-lim, lim], i / 2)  # type: ignore


def plot_results_multi_objective_2D(
    problem: Benchmark,
    individuals_1: NDArray[np.float64],
    individuals_2: NDArray[np.float64],
    title_1: str,
    title_2: str,
    num_objectives: int,
    args: dict[str, Any],
):
    fig = plt.figure(args["fig_title"] + " (initial and final population)")
    lim = None
    axes = []
    for objective in range(num_objectives):
        ax1 = fig.add_subplot(num_objectives, 2, 2 * objective + 1, aspect="equal")
        ax1.plot(individuals_1[:, 0], individuals_1[:, 1], ".b", markersize=7)
        if lim is None:
            lim = max(list(map(abs, ax1.get_xlim())) + list(map(abs, ax1.get_ylim())))
        else:
            lim = max(
                [lim] + list(map(abs, ax1.get_xlim())) + list(map(abs, ax1.get_ylim()))
            )

        ax2 = fig.add_subplot(num_objectives, 2, 2 * objective + 2, aspect="equal")
        ax2.plot(individuals_2[:, 0], individuals_2[:, 1], ".b", markersize=7)
        lim = max(
            [lim] + list(map(abs, ax2.get_xlim())) + list(map(abs, ax2.get_ylim()))
        )
        ax1.set_title(title_1)
        ax2.set_title(title_2)
        axes.append(ax1)
        axes.append(ax2)

    for i, ax in enumerate(axes):
        ax.set_xlim(-lim, lim)  # type: ignore
        ax.set_ylim(-lim, lim)  # type: ignore
        ax.set_xlabel("x0")
        ax.set_ylabel("x1")
        ax.locator_params(nbins=5)
        objective = i / 2
        c = plot_multi_objective_2D(ax, problem, [-lim, lim], objective)  # type: ignore
        if i % 2 == 0:
            cax = fig.add_axes(
                (
                    0.85,
                    (num_objectives - objective - 1) * (0.85 / num_objectives) + 0.12,
                    0.05,
                    0.6 / num_objectives,
                )
            )
            colorbar_ = plt.colorbar(c, cax=cax)
            colorbar_.ax.set_ylabel("Objective " + str(objective + 1))

    fig.subplots_adjust(right=0.8)


def plot_results_multi_objective_PF(individuals: list[Individual], title: str):
    num_objectives: int = len(individuals[0].fitness)  # type: ignore

    if num_objectives < 2:
        pass
    elif num_objectives == 2:
        plt.figure(title)
        plt.plot(
            [guy.fitness[0] for guy in individuals],  # type: ignore
            [guy.fitness[1] for guy in individuals],  # type: ignore
            ".b",
            markersize=7,
        )
        plt.xlabel("f0")
        plt.ylabel("f1")
    else:
        # Creates two subplots and unpacks the output array immediately
        f, axes = plt.subplots(
            num_objectives, num_objectives, sharex="col", sharey="row"
        )
        for i in range(num_objectives):
            for j in range(num_objectives):
                axes[i, j].plot(  # type: ignore
                    [guy.fitness[j] for guy in individuals],  # type: ignore
                    [guy.fitness[i] for guy in individuals],  # type: ignore
                    ".b",
                    markersize=7,
                )
                axes[i, j].set_xlabel("f" + str(j))  # type: ignore
                axes[i, j].set_ylabel("f" + str(i))  # type: ignore
        f.subplots_adjust(hspace=0.30)
        f.subplots_adjust(wspace=0.30)


"""
    the original plot_observer
"""


def plot_observer(
    population: list[Individual],
    num_generations: int,
    num_evaluations: int,
    args: dict[str, Any],
) -> None:
    """Plot the output of the evolutionary computation as a graph.

    This function plots the performance of the EC as a line graph
    using matplotlib and numpy. The graph consists of a blue line
    representing the best fitness, a green line representing the
    average fitness, and a red line representing the median fitness.
    It modifies the keyword arguments variable 'args' by including an
    entry called 'plot_data'.

    If this observer is used, the calling script should also import
    the matplotlib library and should end the script with::

    matplotlib.pyplot.show()

    Otherwise, the program may generate a runtime error.

    .. note::

    This function makes use of the matplotlib and numpy libraries.

    .. Arguments:
    population -- the population of Individuals
    num_generations -- the number of elapsed generations
    num_evaluations -- the number of candidate solution evaluations
    args -- a dictionary of keyword arguments

    """

    stats = inspyred.ec.analysis.fitness_statistics(population)
    best_fitness = stats["best"]
    worst_fitness = stats["worst"]
    median_fitness = stats["median"]
    average_fitness = stats["mean"]

    if isinstance(population[0].fitness, args["class"]):
        candidates: NDArray[np.float64] = np.asarray(
            [guy.candidate for guy in population]
        )
        fitnesses = [
            guy.fitness for guy in single_objective_evaluator(candidates, args)
        ]
        problem = args["problem"]
        if problem.maximize:
            best_fitness = max(fitnesses)
            worst_fitness = min(fitnesses)
        else:
            best_fitness = min(fitnesses)
            worst_fitness = max(fitnesses)
        median_fitness = np.median(fitnesses)
        average_fitness = np.mean(fitnesses)

    colors = ["black", "blue", "green", "red"]
    labels = ["average", "median", "best", "worst"]
    data = []
    if num_generations == 0:
        f, ax = plt.subplots(1, 1, figsize=(12, 8))
        f.suptitle(args["fig_title"] + " (fitness trend)")
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
            (line,) = ax.plot(data[0], data[i + 1], color=colors[i], label=labels[i])  # type: ignore
            lines.append(line)
        args["plot_data"] = data
        args["plot_lines"] = lines
        ax.set_xlabel("Evaluations")
        ax.set_ylabel("Fitness")
    else:
        data = args["plot_data"]
        data[0].append(num_evaluations)
        data[1].append(average_fitness)
        data[2].append(median_fitness)
        data[3].append(best_fitness)
        data[4].append(worst_fitness)
        lines = args["plot_lines"]
        for i, line in enumerate(lines):
            line.set_xdata(numpy.array(data[0]))
            line.set_ydata(numpy.array(data[i + 1]))
        args["plot_data"] = data
        args["plot_lines"] = lines
    ymin = min([min(d) for d in data[1:]])
    ymax = max([max(d) for d in data[1:]])
    yrange = ymax - ymin
    plt.xlim((0, num_evaluations))
    plt.ylim((ymin - 0.1 * yrange, ymax + 0.1 * yrange))
    plt.draw()
    plt.legend()
    # plt.show()


"""
    ann plotting utils
"""


def plot_stats(
    statistics: StatisticsReporter,
    ylog: bool = False,
    view: bool = True,
    filename: Optional[str] = None,
) -> None:
    """Plots the population's average and best fitness."""

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    f, ax = plt.subplots(1, 1, figsize=(12, 8))
    f.suptitle("NEAT (Population's average/std. dev and best fitness)")

    ax.plot(generation, avg_fitness, "b-", label="average")
    ax.plot(generation, avg_fitness - stdev_fitness, "g-.", label="-1 sd")
    ax.plot(generation, avg_fitness + stdev_fitness, "g-.", label="+1 sd")
    ax.plot(generation, best_fitness, "r-", label="best")

    # plt.title("Population's average/std. dev and best fitness")
    ax.set_xlabel("Generations")
    ax.set_ylabel("Fitness")
    ax.grid()
    ax.legend(loc="best")
    if ylog:
        plt.gca().set_yscale("symlog")

    if filename is not None:
        plt.savefig(filename)

    if view:
        plt.show()


def plot_spikes(
    spikes: list[tuple[float, float, float, float, float]],
    view: bool = True,
    filename: Optional[str] = None,
    title: Optional[str] = None,
):
    """Plots the trains for a single spiking neuron."""
    t_values, I_values, v_values, u_values, f_values = zip(*spikes)

    ax: list[Axes]
    fig, ax = plt.subplots(4, 1, figsize=(12, 8))  # type: ignore
    ax[0].set_ylabel("Membrane Potential (V)")
    ax[0].set_xlabel("Time (in ms)")
    ax[0].grid()
    ax[0].plot(t_values, v_values, "g-")

    ax[1].set_ylabel("Fired")
    ax[1].set_xlabel("Time (in ms)")
    ax[1].grid()
    ax[1].plot(t_values, f_values, "r-")

    ax[2].set_ylabel("Recovery (u)")
    ax[2].set_xlabel("Time (in ms)")
    ax[2].grid()
    ax[2].plot(t_values, u_values, "r-")

    ax[3].set_ylabel("Current (I)")
    ax[3].set_xlabel("Time (in ms)")
    ax[3].grid()
    ax[3].plot(t_values, I_values, "r-o")

    if title is None:
        fig.suptitle("Izhikevich's spiking neuron model")
    else:
        fig.suptitle("Izhikevich's spiking neuron model ({0!s})".format(title))

    if filename is not None:
        fig.savefig(filename)

    if view:
        fig.show()


def plot_species(
    statistics: StatisticsReporter, view: bool = True, filename: Optional[str] = None
) -> None:
    """Visualizes speciation throughout evolution."""
    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    fig = plt.figure("NEAT (speciation)")
    ax = fig.add_subplot(111)
    ax.stackplot(range(num_generations), *curves)

    # plt.title("Speciation")
    ax.set_ylabel("Size per Species")
    ax.set_xlabel("Generations")

    if filename is not None:
        fig.savefig(filename)

    if view:
        fig.show()


def draw_net(
    config: Config,
    genome: Any,
    view: bool = True,
    filename: Optional[str] = None,
    node_names: Optional[dict[Any, Any]] = None,
    show_disabled: bool = True,
    prune_unused: bool = False,
    node_colors: Optional[dict[Any, Any]] = None,
    fmt: str = "png",
) -> Any:
    """Receives a genome and draws a neural network with arbitrary topology."""
    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {"shape": "circle", "fontsize": "9", "height": "0.2", "width": "0.2"}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {"style": "filled", "shape": "box"}
        input_attrs["fillcolor"] = node_colors.get(k, "lightgray")
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {"style": "filled"}
        node_attrs["fillcolor"] = node_colors.get(k, "lightblue")

        dot.node(name, _attributes=node_attrs)

    if prune_unused:
        connections = set()
        for cg in genome.connections.values():
            if cg.enabled or show_disabled:
                connections.add((cg.in_node_id, cg.out_node_id))

        used_nodes = copy.copy(outputs)
        pending = copy.copy(outputs)
        while pending:
            new_pending = set()
            for a, b in connections:
                if b in pending and a not in used_nodes:
                    new_pending.add(a)
                    used_nodes.add(a)
            pending = new_pending
    else:
        used_nodes = set(genome.nodes.keys())

    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {"style": "filled", "fillcolor": node_colors.get(n, "white")}
        dot.node(str(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            # if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = "solid" if cg.enabled else "dotted"
            color = "green" if cg.weight > 0 else "red"
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(
                a, b, _attributes={"style": style, "color": color, "penwidth": width}
            )

    dot.render(filename, view=view)

    return dot


"""

"""


def plotTrends(logbook: Logbook, name: Optional[str] = None):
    gen = logbook.select("gen")
    fit_min = logbook.select("min")
    fit_max = logbook.select("max")
    fit_avg = logbook.select("avg")
    fit_std = logbook.select("std")

    fig = plt.figure("Genetic Programming (fitness trend)")
    ax1 = fig.add_subplot(111)
    ax1.plot(gen, fit_min, label="Min")
    ax1.plot(gen, fit_max, label="Max")
    ax1.errorbar(gen, fit_avg, yerr=fit_std, label="Avg")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    ax1.set_xlim(0, len(gen) - 1)
    ax1.legend()
    if name is not None:
        plt.savefig(name)
    plt.show()


def plotTree(
    nodes: list[pgv.Node],
    edges: list[tuple[pgv.Node, pgv.Node]],
    labels: dict[pgv.Node, str],
    name: str,
    folder: str,
):
    plt.figure("GP (best tree)")
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
    nx.draw_networkx_nodes(g, pos)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels)
    plt.savefig(folder + "/" + "tree_" + name + ".png")
