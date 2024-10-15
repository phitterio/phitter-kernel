import math
import random

import numpy as np
from graphviz import Digraph
from IPython.display import display

import phitter


class ProcessSimulation:
    def __init__(self) -> None:

        self.process_prob_distr = dict()
        self.branches = dict()
        self.order = dict()
        self.number_of_products = dict()
        self.process_positions = dict()
        self.next_process = dict()
        self.probability_distribution = (
            phitter.continuous.CONTINUOUS_DISTRIBUTIONS
            | phitter.discrete.DISCRETE_DISTRIBUTIONS
        )

    def add_process(
        self,
        prob_distribution: str,
        parameters: dict,
        process_id: str,
        number_of_products: int = 1,
        new_branch: bool = False,
        previous_ids: list[str] = None,
    ) -> None:
        """Add element to the simulation

        Args:
            prob_distribution (str): Probability distribution to be used. You can use one of the following: 'alpha', 'arcsine', 'argus', 'beta', 'beta_prime', 'beta_prime_4p', 'bradford', 'burr', 'burr_4p', 'cauchy', 'chi_square', 'chi_square_3p', 'dagum', 'dagum_4p', 'erlang', 'erlang_3p', 'error_function', 'exponential', 'exponential_2p', 'f', 'fatigue_life', 'folded_normal', 'frechet', 'f_4p', 'gamma', 'gamma_3p', 'generalized_extreme_value', 'generalized_gamma', 'generalized_gamma_4p', 'generalized_logistic', 'generalized_normal', 'generalized_pareto', 'gibrat', 'gumbel_left', 'gumbel_right', 'half_normal', 'hyperbolic_secant', 'inverse_gamma', 'inverse_gamma_3p', 'inverse_gaussian', 'inverse_gaussian_3p', 'johnson_sb', 'johnson_su', 'kumaraswamy', 'laplace', 'levy', 'loggamma', 'logistic', 'loglogistic', 'loglogistic_3p', 'lognormal', 'maxwell', 'moyal', 'nakagami', 'non_central_chi_square', 'non_central_f', 'non_central_t_student', 'normal', 'pareto_first_kind', 'pareto_second_kind', 'pert', 'power_function', 'rayleigh', 'reciprocal', 'rice', 'semicircular', 'trapezoidal', 'triangular', 't_student', 't_student_3p', 'uniform', 'weibull', 'weibull_3p', 'bernoulli', 'binomial', 'geometric', 'hypergeometric', 'logarithmic', 'negative_binomial', 'poisson'.
            parameters (dict): Parameters of the probability distribution.
            process_id (str):  Unique name of the process to be simulated
            number_of_products (int ,optional): Number of elements that are need to simulate in that stage. Value has to be greater than 0. Defaults equals to 1.
            new_branch (bool ,optional): Required if you want to start a new process that does not have previous processes. You cannot use this parameter at the same time with "previous_id". Defaults to False.
            previous_id (list[str], optional): Required if you have previous processes that are before this process. You cannot use this parameter at the same time with "new_branch". Defaults to None.
        """

        if prob_distribution not in self.probability_distribution.keys():
            raise ValueError(
                f"""Please select one of the following probability distributions: '{"', '".join(self.probability_distribution.keys())}'."""
            )
        else:
            if process_id not in self.order.keys():
                if number_of_products >= 1:
                    if new_branch == True and previous_ids != None:
                        raise ValueError(
                            f"""You cannot select 'new_branch' is equals to True if 'previous_id' is not empty. OR you cannot add 'previous_ids' if 'new_branch' is equals to True."""
                        )
                    else:
                        if new_branch == True:
                            branch_id = len(self.branches)
                            self.branches[branch_id] = process_id
                            self.order[process_id] = branch_id
                            self.number_of_products[process_id] = number_of_products
                            self.process_prob_distr[process_id] = (
                                self.probability_distribution[prob_distribution](
                                    parameters
                                )
                            )
                            self.next_process[process_id] = 0
                        elif previous_ids != None and all(
                            id in self.order.keys() for id in previous_ids
                        ):
                            self.order[process_id] = previous_ids
                            self.number_of_products[process_id] = number_of_products
                            self.process_prob_distr[process_id] = (
                                self.probability_distribution[prob_distribution](
                                    parameters
                                )
                            )
                            self.next_process[process_id] = 0
                            for prev_id in previous_ids:
                                self.next_process[prev_id] += 1
                        else:
                            raise ValueError(
                                f"""Please create a new_brach == True if you need a new process or specify the previous process/processes (previous_ids) that are before this one. Processes that have been added: '{"', '".join(self.order.keys())}'."""
                            )
                else:
                    raise ValueError(
                        f"""You must add number_of_products grater or equals than 1."""
                    )
            else:
                raise ValueError(
                    f"""You need to create diferent process_id for each process, '{process_id}' already exists."""
                )

    def run(self, number_of_simulations: int = 1) -> list[float]:
        """Simulation of the described process

        Args:
            number_of_simulations (int, optional): Number of simulations of the process that you want to do. Defaults to 1.

        Returns:
            list[float]: Results of every simulation requested
        """
        simulation_result = list()
        for simulation in range(number_of_simulations):
            simulation_partial_result = dict()
            simulation_accumulative_result = dict()
            for key in self.branches.keys():
                partial_result = 0
                for _ in range(self.number_of_products[self.branches[key]]):
                    partial_result += self.process_prob_distr[self.branches[key]].ppf(
                        random.random()
                    )
                simulation_partial_result[self.branches[key]] = partial_result
                simulation_accumulative_result[self.branches[key]] = (
                    simulation_partial_result[self.branches[key]]
                )
            for key in self.process_prob_distr.keys():
                if isinstance(self.order[key], list):
                    partial_result = 0
                    for _ in range(self.number_of_products[key]):
                        partial_result += self.process_prob_distr[key].ppf(
                            random.random()
                        )
                    simulation_partial_result[key] = partial_result
                    simulation_accumulative_result[key] = (
                        simulation_partial_result[key]
                        + simulation_accumulative_result[
                            max(self.order[key], key=simulation_accumulative_result.get)
                        ]
                    )
            simulation_result.append(
                simulation_accumulative_result[
                    max(
                        simulation_accumulative_result.keys(),
                        key=simulation_accumulative_result.get,
                    )
                ]
            )
        return simulation_result

    def run_confidence_interval(
        self,
        confidence_level: float = 0.95,
        number_of_simulations: int = 1,
        replications: int = 30,
    ) -> tuple[float]:
        """Generates a confidence interval for the replications of the requested number of simulations.

        Args:
            confidence_level (float, optional): Confidence required of the interval. Defaults to 0.95.
            number_of_simulations (int, optional): Number of simulations that are going to be run in each replication. Defaults to 1.
            replications (int, optional): Number of samples needed. Defaults to 30.

        Returns:
            tuple[float]: Returns the lower bound, average, upper bound and standard deviation of the confidence interval
        """
        # Simulate with replications
        average_results_simulations = [
            np.mean(self.run(number_of_simulations)) for _ in range(replications)
        ]
        # Confidence Interval
        ## Sample standard deviation
        standar_deviation = np.std(average_results_simulations, ddof=1)
        standard_error = standar_deviation / math.sqrt(replications)
        average = np.mean(average_results_simulations)
        normal_standard = phitter.continuous.NORMAL({"mu": 0, "sigma": 1})
        z = normal_standard.ppf((1 + confidence_level) / 2)
        ## Confidence Interval
        upper_bound = average + (z * standard_error)
        lower_bound = average - (z * standard_error)
        # Return confidence interval
        return lower_bound, average, upper_bound, standar_deviation

    def process_graph(
        self, graph_direction: str = "LR", save_graph_pdf: bool = False
    ) -> None:
        """Generates the graph of the process

        Args:
            graph_direction (str, optional): You can show the graph in two ways: 'LR' left to right OR 'TB' top to bottom. Defaults to 'LR'.
            save_graph_pdf (bool, optional): You can save the process graph in a PDF file. Defaults to False.
        """
        graph = Digraph(comment="Simulation Process Steb-by-Step")

        for node in set(self.order.keys()):
            print(node)
            if isinstance(self.order[node], int):
                graph.node(
                    node, node, shape="circle", style="filled", fillcolor="lightgreen"
                )
            elif self.next_process[node] == 0:
                graph.node(
                    node,
                    node,
                    shape="doublecircle",
                    style="filled",
                    fillcolor="lightblue",
                )
            else:
                graph.node(node, node, shape="box")

        for node in set(self.order.keys()):
            if isinstance(self.order[node], list):
                for previous_node in self.order[node]:
                    graph.edge(
                        previous_node,
                        node,
                        label=str(self.number_of_products[previous_node]),
                        fontsize="10",
                    )

        if graph_direction == "TB":
            graph.attr(rankdir="TB")
        else:
            graph.attr(rankdir="LR")

        if save_graph_pdf:
            graph.render("Simulation Process Steb-by-Step", view=True)

        display(graph)
