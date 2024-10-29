import math
import random

import pandas as pd
from graphviz import Digraph
from IPython.display import display

import phitter


class ProcessSimulation:
    def __init__(self) -> None:
        """Build the object Process Simulation"""

        self.process_prob_distr = dict()
        self.branches = dict()
        self.order = dict()
        self.number_of_products = dict()
        self.process_positions = dict()
        self.next_process = dict()
        self.probability_distribution = phitter.continuous.CONTINUOUS_DISTRIBUTIONS | phitter.discrete.DISCRETE_DISTRIBUTIONS
        self.servers = dict()

        self.simulation_result = dict()

    def __str__(self) -> str:
        """Print dataset

        Returns:
            str: Dataframe in str mode
        """
        sim = pd.DataFrame(self.simulation_result)
        return sim.to_string()

    def _repr_html_(self) -> pd.DataFrame:
        """Print DataFrames in jupyter notebooks

        Returns:
            pd.DataFrame: Simulation result
        """
        sim = pd.DataFrame(self.simulation_result)
        return sim._repr_html_()

    def __getitem__(self, key):
        sim = pd.DataFrame(self.simulation_result)
        return sim[key]

    def add_process(
        self,
        prob_distribution: str,
        parameters: dict,
        process_id: str,
        number_of_products: int = 1,
        number_of_servers: int = 1,
        new_branch: bool = False,
        previous_ids: list[str] = None,
    ) -> None:
        """Add element to the simulation

        Args:
            prob_distribution (str): Probability distribution to be used. You can use one of the following: 'alpha', 'arcsine', 'argus', 'beta', 'beta_prime', 'beta_prime_4p', 'bradford', 'burr', 'burr_4p', 'cauchy', 'chi_square', 'chi_square_3p', 'dagum', 'dagum_4p', 'erlang', 'erlang_3p', 'error_function', 'exponential', 'exponential_2p', 'f', 'fatigue_life', 'folded_normal', 'frechet', 'f_4p', 'gamma', 'gamma_3p', 'generalized_extreme_value', 'generalized_gamma', 'generalized_gamma_4p', 'generalized_logistic', 'generalized_normal', 'generalized_pareto', 'gibrat', 'gumbel_left', 'gumbel_right', 'half_normal', 'hyperbolic_secant', 'inverse_gamma', 'inverse_gamma_3p', 'inverse_gaussian', 'inverse_gaussian_3p', 'johnson_sb', 'johnson_su', 'kumaraswamy', 'laplace', 'levy', 'loggamma', 'logistic', 'loglogistic', 'loglogistic_3p', 'lognormal', 'maxwell', 'moyal', 'nakagami', 'non_central_chi_square', 'non_central_f', 'non_central_t_student', 'normal', 'pareto_first_kind', 'pareto_second_kind', 'pert', 'power_function', 'rayleigh', 'reciprocal', 'rice', 'semicircular', 'trapezoidal', 'triangular', 't_student', 't_student_3p', 'uniform', 'weibull', 'weibull_3p', 'bernoulli', 'binomial', 'geometric', 'hypergeometric', 'logarithmic', 'negative_binomial', 'poisson'.
            parameters (dict): Parameters of the probability distribution.
            process_id (str):  Unique name of the process to be simulated
            number_of_products (int ,optional): Number of elements that are needed to simulate in that stage. Value has to be greater than 0. Defaults equals to 1.
            number_of_servers (int, optional): Number of servers that process has and are needed to simulate in that stage. Value has to be greater than 0. Defaults equals to 1.
            new_branch (bool ,optional): Required if you want to start a new process that does not have previous processes. You cannot use this parameter at the same time with "previous_id". Defaults to False.
            previous_id (list[str], optional): Required if you have previous processes that are before this process. You cannot use this parameter at the same time with "new_branch". Defaults to None.
        """
        # Verify if the probability is created in phitter
        if prob_distribution not in self.probability_distribution.keys():
            raise ValueError(f"""Please select one of the following probability distributions: '{"', '".join(self.probability_distribution.keys())}'.""")
        else:
            # Verify unique id name for each process
            if process_id not in self.order.keys():
                # Verify that at least is one element needed for the simulation in that stage
                if number_of_products >= 1:
                    # Verify if the number of servers is greater or equals than 1
                    if number_of_servers >= 1:
                        # Verify that if you create a new branch, it's impossible to have a previous id (or preceding process). One of those is incorrect
                        if new_branch == True and previous_ids != None:
                            raise ValueError(f"""You cannot select 'new_branch' is equals to True if 'previous_id' is not empty. OR you cannot add 'previous_ids' if 'new_branch' is equals to True.""")
                        else:
                            # If it is a new branch then initialize all the needed paramters
                            if new_branch == True:
                                branch_id = len(self.branches)
                                self.branches[branch_id] = process_id
                                self.order[process_id] = branch_id
                                self.number_of_products[process_id] = number_of_products
                                self.servers[process_id] = number_of_servers
                                self.process_prob_distr[process_id] = self.probability_distribution[prob_distribution](parameters)
                                self.next_process[process_id] = 0
                                # Create id of that process in the simulation result
                                self.simulation_result[process_id] = []

                            # If it is NOT a new branch then initialize all the needed paramters
                            elif previous_ids != None and all(id in self.order.keys() for id in previous_ids):
                                self.order[process_id] = previous_ids
                                self.number_of_products[process_id] = number_of_products
                                self.servers[process_id] = number_of_servers
                                self.process_prob_distr[process_id] = self.probability_distribution[prob_distribution](parameters)
                                self.next_process[process_id] = 0
                                # Create id of that process in the simulation result
                                self.simulation_result[process_id] = []
                                for prev_id in previous_ids:
                                    self.next_process[prev_id] += 1
                            # if something is incorrect then raise an error
                            else:
                                raise ValueError(
                                    f"""Please create a new_brach == True if you need a new process or specify the previous process/processes (previous_ids) that are before this one. Processes that have been added: '{"', '".join(self.order.keys())}'."""
                                )
                    else:
                        raise ValueError(f"""You must add number_of_servers grater or equals than 1.""")
                else:
                    raise ValueError(f"""You must add number_of_products grater or equals than 1.""")
            else:
                raise ValueError(f"""You need to create diferent process_id for each process, '{process_id}' already exists.""")

    def run(self, number_of_simulations: int = 1) -> list[float]:
        """Simulation of the described process

        Args:
            number_of_simulations (int, optional): Number of simulations of the process that you want to do. Defaults to 1.

        Returns:
            list[float]: Results of every simulation requested
        """

        self.simulation_result = {key: list() for key in self.simulation_result.keys()}

        # Create simulation list
        simulation_result = list()
        # Start all possible simulations
        for simulation in range(number_of_simulations):
            # Create dictionaries for identifing processes
            simulation_partial_result = dict()
            simulation_accumulative_result = dict()

            # For every single "new branch" process
            for key in self.branches.keys():
                partial_result = 0
                # If there is only one server in that process
                if self.servers[self.branches[key]] == 1:
                    # Simulate the time it took to create each product needed
                    for _ in range(self.number_of_products[self.branches[key]]):
                        partial_result += self.process_prob_distr[self.branches[key]].ppf(random.random())
                    # Add all simulation time according to the time it took to create all products in that stage
                    simulation_partial_result[self.branches[key]] = partial_result
                    # Add this partial result to see the average time of this specific process
                    self.simulation_result[self.branches[key]].append(simulation_partial_result[self.branches[key]])
                    # Because we are simulating the "new branch" or first processes, accumulative it's the same as partial result
                    simulation_accumulative_result[self.branches[key]] = simulation_partial_result[self.branches[key]]
                # If there are more than one servers in that process
                else:
                    # Simulate the time it took to create each product needed
                    products_times = [self.process_prob_distr[self.branches[key]].ppf(random.random()) for _ in range(self.number_of_products[self.branches[key]])]

                    # Initialize dictionary
                    servers_dictionary = {server: 0 for server in range(self.servers[self.branches[key]])}

                    # Organize times according to the number of machines you have
                    for product in products_times:
                        # Identify server with the shortest time of all
                        min_server_time = min(servers_dictionary, key=servers_dictionary.get)
                        # Add product time to that server
                        servers_dictionary[min_server_time] += product

                    # Identify the "partial result" as the maximum time of all servers to create all products
                    partial_result = max(servers_dictionary.values())

                    # Add all simulation time according to the time it took to create all products in that stage
                    simulation_partial_result[self.branches[key]] = partial_result
                    # Add this partial result to see the average time of this specific process
                    self.simulation_result[self.branches[key]].append(simulation_partial_result[self.branches[key]])
                    # Because we are simulating the "new branch" or first processes, accumulative it's the same as partial result
                    simulation_accumulative_result[self.branches[key]] = simulation_partial_result[self.branches[key]]

            # For every process
            for key in self.process_prob_distr.keys():
                # Only consider the ones that are not "New Branches"
                if isinstance(self.order[key], list):
                    partial_result = 0
                    # If there is only one server in that process
                    if self.servers[key] == 1:
                        # Simulate all products time
                        for _ in range(self.number_of_products[key]):
                            partial_result += self.process_prob_distr[key].ppf(random.random())
                        # Save partial result
                        simulation_partial_result[key] = partial_result
                        # Add this partial result to see the average time of this specific process
                        self.simulation_result[key].append(simulation_partial_result[key])
                        # Accumulate this partial result plus the previous processes of this process
                        simulation_accumulative_result[key] = (
                            simulation_partial_result[key]
                            + simulation_accumulative_result[
                                max(
                                    self.order[key],
                                    key=simulation_accumulative_result.get,
                                )
                            ]
                        )
                    # If there are more than one servers in that process
                    else:
                        # Simulate the time it took to create each product needed
                        products_times = [self.process_prob_distr[key].ppf(random.random()) for _ in range(self.number_of_products[key])]

                        # Initialize dictionary
                        servers_dictionary = {server: 0 for server in range(self.servers[key])}

                        # Organize times according to the number of machines you have
                        for product in products_times:
                            # Identify server with the shortest time of all
                            min_server_time = min(servers_dictionary, key=servers_dictionary.get)
                            # Add product time to that server
                            servers_dictionary[min_server_time] += product

                        # Identify the "partial result" as the maximum time of all servers to create all products
                        partial_result = max(servers_dictionary.values())

                        # Save partial result
                        simulation_partial_result[key] = partial_result
                        # Add this partial result to see the average time of this specific process
                        self.simulation_result[key].append(simulation_partial_result[key])
                        # Accumulate this partial result plus the previous processes of this process
                        simulation_accumulative_result[key] = (
                            simulation_partial_result[key]
                            + simulation_accumulative_result[
                                max(
                                    self.order[key],
                                    key=simulation_accumulative_result.get,
                                )
                            ]
                        )

            # Save the max time of the simulation
            simulation_result.append(
                simulation_accumulative_result[
                    max(
                        simulation_accumulative_result.keys(),
                        key=simulation_accumulative_result.get,
                    )
                ]
            )
        self.simulation_result["Total Simulation Time"] = simulation_result
        return pd.DataFrame(self.simulation_result)

    def simulation_metrics(self) -> pd.DataFrame:
        """Here you can find the average time per process and standard deviation

        Returns:
            pd.DataFrame: Average and Standard deviation
        """
        # Use simulation results
        df = pd.DataFrame(self.simulation_result)

        if len(df) == 0:
            raise ValueError("You need to run the simulation first")

        # Calculate all metrics
        metrics_dict_1 = {f"Avg. {column}": df[column].mean() for column in df.columns}
        metrics_dict_2 = {f"Std. Dev. {column}": df[column].std() for column in df.columns}
        metrics_dict = metrics_dict_1 | metrics_dict_2

        # Create result dataframe
        metrics = pd.DataFrame.from_dict(metrics_dict, orient="index").rename(columns={0: "Value"})

        metrics.index.name = "Metrics"

        return metrics.reset_index()

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
        # Initializa variables
        tot_metrics = pd.DataFrame()
        for _ in range(replications):
            # Run simulation
            self.run(number_of_simulations)
            # Save metrics and probabilities
            metrics_summary = self.simulation_metrics()
            # Concat previous results with current results
            tot_metrics = pd.concat([tot_metrics, metrics_summary])

        # Confidence Interval
        std__2 = tot_metrics.groupby(["Metrics"]).std()
        mean__2 = tot_metrics.groupby(["Metrics"]).mean()

        standard_error = std__2 / math.sqrt(replications)
        normal_standard = phitter.continuous.NORMAL({"mu": 0, "sigma": 1})
        z = normal_standard.ppf((1 + confidence_level) / 2)
        ## Confidence Interval
        avg__2 = mean__2.copy()
        lower_bound = (mean__2 - (z * standard_error)).copy().rename(columns={"Value": "LB - Value"})
        upper_bound = (mean__2 + (z * standard_error)).copy().rename(columns={"Value": "UB - Value"})
        avg__2 = avg__2.rename(columns={"Value": "AVG - Value"})
        tot_metrics_interval = pd.concat([lower_bound, avg__2, upper_bound], axis=1)
        tot_metrics_interval = tot_metrics_interval[["LB - Value", "AVG - Value", "UB - Value"]]
        # Return confidence interval
        return tot_metrics_interval.reset_index()

    def process_graph(self, graph_direction: str = "LR", save_graph_pdf: bool = False) -> None:
        """Generates the graph of the process

        Args:
            graph_direction (str, optional): You can show the graph in two ways: 'LR' left to right OR 'TB' top to bottom. Defaults to 'LR'.
            save_graph_pdf (bool, optional): You can save the process graph in a PDF file. Defaults to False.
        """
        # Create graph instance
        graph = Digraph(comment="Simulation Process Steb-by-Step")

        # Add all nodes
        for node in set(self.order.keys()):
            # Identify if this is a "New branch"
            if isinstance(self.order[node], int):
                graph.node(
                    f"{node} - {self.servers[node]} server(s)",
                    f"{node} - {self.servers[node]} server(s)",
                    shape="circle",
                    style="filled",
                    fillcolor="lightgreen",
                )
            elif self.next_process[node] == 0:
                graph.node(
                    f"{node} - {self.servers[node]} server(s)",
                    f"{node} - {self.servers[node]} server(s)",
                    shape="doublecircle",
                    style="filled",
                    fillcolor="lightblue",
                )
            else:
                graph.node(
                    f"{node} - {self.servers[node]} server(s)",
                    f"{node} - {self.servers[node]} server(s)",
                    shape="box",
                )

        for node in set(self.order.keys()):
            # Identify if this is a "Previous id"
            if isinstance(self.order[node], list):
                for previous_node in self.order[node]:
                    graph.edge(
                        f"{previous_node} - {self.servers[previous_node]} server(s)",
                        f"{node} - {self.servers[node]} server(s)",
                        label=str(self.number_of_products[previous_node]),
                        fontsize="10",
                    )

        # Graph direction
        if graph_direction == "TB":
            graph.attr(rankdir="TB")
        else:
            graph.attr(rankdir="LR")

        # If needed save graph in pdf
        if save_graph_pdf:
            graph.render("Simulation Process Steb-by-Step", view=True)

        # Show graph
        display(graph)
