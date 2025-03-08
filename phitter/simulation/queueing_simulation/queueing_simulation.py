import phitter
import pandas as pd
import random
import numpy as np
import collections

import math

from phitter.simulation.own_distribution import OwnDistributions


class QueueingSimulation:
    def __init__(
        self,
        a: str,
        a_parameters: dict,
        s: str,
        s_parameters: dict,
        c: int,
        k: float = float("inf"),
        n: float = float("inf"),
        d: str = "FIFO",
        pbs_distribution: str | None = None,
        pbs_parameters: dict | None = None,
    ) -> None:
        """Simulation of any queueing model.

        Args:
            a (str): Arrival time distribution (Arrival). Distributions that can be used: 'alpha', 'arcsine', 'argus', 'beta', 'beta_prime', 'beta_prime_4p', 'bradford', 'burr', 'burr_4p', 'cauchy', 'chi_square', 'chi_square_3p', 'dagum', 'dagum_4p', 'erlang', 'erlang_3p', 'error_function', 'exponential', 'exponential_2p', 'f', 'fatigue_life', 'folded_normal', 'frechet', 'f_4p', 'gamma', 'gamma_3p', 'generalized_extreme_value', 'generalized_gamma', 'generalized_gamma_4p', 'generalized_logistic', 'generalized_normal', 'generalized_pareto', 'gibrat', 'gumbel_left', 'gumbel_right', 'half_normal', 'hyperbolic_secant', 'inverse_gamma', 'inverse_gamma_3p', 'inverse_gaussian', 'inverse_gaussian_3p', 'johnson_sb', 'johnson_su', 'kumaraswamy', 'laplace', 'levy', 'loggamma', 'logistic', 'loglogistic', 'loglogistic_3p', 'lognormal', 'maxwell', 'moyal', 'nakagami', 'non_central_chi_square', 'non_central_f', 'non_central_t_student', 'normal', 'pareto_first_kind', 'pareto_second_kind', 'pert', 'power_function', 'rayleigh', 'reciprocal', 'rice', 'semicircular', 'trapezoidal', 'triangular', 't_student', 't_student_3p', 'uniform', 'weibull', 'weibull_3p', 'bernoulli', 'binomial', 'geometric', 'hypergeometric', 'logarithmic', 'negative_binomial', 'poisson'.
            a_parameters (dict): All the needed parameters to use the probability distribution of the arrival time.
            s (str): Service time distribution (Service). Distributions that can be used: 'alpha', 'arcsine', 'argus', 'beta', 'beta_prime', 'beta_prime_4p', 'bradford', 'burr', 'burr_4p', 'cauchy', 'chi_square', 'chi_square_3p', 'dagum', 'dagum_4p', 'erlang', 'erlang_3p', 'error_function', 'exponential', 'exponential_2p', 'f', 'fatigue_life', 'folded_normal', 'frechet', 'f_4p', 'gamma', 'gamma_3p', 'generalized_extreme_value', 'generalized_gamma', 'generalized_gamma_4p', 'generalized_logistic', 'generalized_normal', 'generalized_pareto', 'gibrat', 'gumbel_left', 'gumbel_right', 'half_normal', 'hyperbolic_secant', 'inverse_gamma', 'inverse_gamma_3p', 'inverse_gaussian', 'inverse_gaussian_3p', 'johnson_sb', 'johnson_su', 'kumaraswamy', 'laplace', 'levy', 'loggamma', 'logistic', 'loglogistic', 'loglogistic_3p', 'lognormal', 'maxwell', 'moyal', 'nakagami', 'non_central_chi_square', 'non_central_f', 'non_central_t_student', 'normal', 'pareto_first_kind', 'pareto_second_kind', 'pert', 'power_function', 'rayleigh', 'reciprocal', 'rice', 'semicircular', 'trapezoidal', 'triangular', 't_student', 't_student_3p', 'uniform', 'weibull', 'weibull_3p', 'bernoulli', 'binomial', 'geometric', 'hypergeometric', 'logarithmic', 'negative_binomial', 'poisson'.
            s_parameters (dict): All the needed parameters to use the probability distribution of the service time.
            c (int): Number of servers. This represents the total number of service channels available in the system. It indicates how many customers can be served simultaneously, affecting the system's capacity to handle incoming clients and impacting metrics like waiting times and queue lengths.
            k (float, optional): Maximum system capacity. This is the maximum number of customers that the system can accommodate at any given time, including both those in service and those waiting in the queue. It defines the limit beyond which arriving customers are either turned away or blocked from entering the system. Defaults to float("inf").
            n (float, optional): Total population of potential customers. This denotes the overall number of potential customers who might require service from the system. It can be finite or infinite and affects the arrival rates and the modeling of the system, especially in closed queueing networks. Defaults to float("inf").
            d (str, optional): Queue discipline. This describes the rule or policy that determines the order in which customers are served. Common disciplines include First-In-First-Out ("FIFO"), Last-In-First-Out ("LIFO"), priority-based service ("PBS"). The queue discipline impacts waiting times and the overall fairness of the system. Defaults to "FIFO".
            pbs_distribution (str | None, optional): Discrete distribution that identifies the label of the pbs, this parameter can only be used with "d='PBS'". Distributions that can be used: 'own_distribution', 'bernoulli', 'binomial', 'geometric', 'hypergeometric', 'logarithmic', 'negative_binomial', 'poisson'. Defaults to None.
            pbs_parameters (dict | None, optional): Parameters of the discrete distribution that identifies the label of the pbs, this parameter can only be used with "d='PBS'". If it is 'own-distribution' add labels in the following way (example): {0: 0.5, 1: 0.3, 2: 0.2}. Where the "key" corresponds to the label and the "value" the probability whose total sum must add up to 1; "keys" with greater importances are the greaters and always have to be numeric keys. You can add as labels as you need.
        """

        # All phitter probability distributions
        self.__probability_distribution = phitter.continuous.CONTINUOUS_DISTRIBUTIONS | phitter.discrete.DISCRETE_DISTRIBUTIONS

        # Distributions for labels
        self.__pbs_distributions = {"own_distribution": OwnDistributions} | phitter.discrete.DISCRETE_DISTRIBUTIONS
        # Queue discipline
        self.__queue_discipline = ["FIFO", "LIFO", "PBS"]
        # Verify if all variables are correct
        self.__verify_variables(a, s, c, k, n, d, pbs_distribution, pbs_parameters)

        # Saving input variables
        self.__save_a = a
        self.__save_a_params = a_parameters
        self.__save_s = s
        self.__save_s_params = s_parameters

        # Variables input for simualtion
        self.__a = self.__probability_distribution[self.__save_a](self.__save_a_params)
        self.__s = self.__probability_distribution[self.__save_s](self.__save_s_params)
        self.__c = c
        self.__k = k
        self.__n = n
        self.__d = d

        # This variables are assigned later
        self.__simulation_time = 0

        # PBS parameters
        self.__pbs_parameters = pbs_parameters
        self.__pbs_distribution = pbs_distribution
        if d == "PBS":
            self.__label = self.__pbs_distributions[self.__pbs_distribution](self.__pbs_parameters)

        # Simulation results
        self.result_simulation = pd.DataFrame()
        self.__simulation_time = 0
        self.number_probabilities = dict()

    def __str__(self) -> str:
        """Print dataset

        Returns:
            str: Dataframe in str mode
        """
        return self.__result_simulation.to_string()

    def _repr_html_(self) -> pd.DataFrame:
        """Print DataFrames in jupyter notebooks

        Returns:
            pd.DataFrame: Simulation result
        """
        return self.__result_simulation._repr_html_()

    def __getitem__(self, key):
        return self.__result_simulation[key]

    def __verify_variables(
        self,
        a: str,
        s: str,
        c: int,
        k: float,
        n: float,
        d: str,
        pbs_distribution: str,
        pbs_parameters: dict,
    ) -> None:
        """Verify if the variable value is correct

        Args:
            a (str): Arrival time distribution (Arrival).
            s (str): Service time distribution (Service).
            c (int): Number of servers.
            k (float): Maximum system capacity.
            n (float): Total population of potential customers.
            d (str): Queue discipline.
            pbs_distribution (str): Label Distribution. It can only be used if d='PBS' is selected
            pbs_parameters (dict): Parameters of the PBS Distribution. It can only be used if d='PBS' is selected
        """

        # Verify if a and s belong to a actual probability distribution
        if a not in self.__probability_distribution.keys() or s not in self.__probability_distribution.keys():
            raise ValueError(f"""Please select one of the following probability distributions: '{"', '".join(self.__probability_distribution.keys())}'.""")
        # Verify Number of Servers
        if c <= 0:
            raise ValueError(f"""'c' has to be a number and cannot be less or equals than zero.""")
        # Verify Maximum System Capacity
        if k <= 0:
            raise ValueError(f"""'k' has to be a number and cannot be less or equals than zero.""")
        # Verify Total population of potential customers.
        if n <= 0:
            raise ValueError(f"""'n' has to be a number and cannot be less or equals than zero.""")

        # Review if the discipline is in the list
        if d not in self.__queue_discipline:
            raise ValueError(f"""'d' has to be one of the following queue discipline: '{"', '".join(self.__queue_discipline)}'.""")

        # Maximum number of people should be greater or equals to the number of servers
        if k < c:
            raise ValueError(f"""'k' cannot be less than the number of servers (c)""")

        # PBS logic
        if d == "PBS":
            if pbs_distribution != None and pbs_parameters != None:
                # Review if the selected distribution was created
                if pbs_distribution not in self.__pbs_distributions:
                    raise ValueError(f"""You should select one of the following distributions: {self.__pbs_distributions}""")
            # Review if PBS is selected if the distribution and parameters exits
            elif pbs_distribution == None and pbs_parameters == None:
                raise ValueError(f"""You must include 'pbs_distribution' and 'pbs_parameters' if you want to use 'PBS'.""")
        # You can only use this two parameters if PBS is selected
        elif d != "PBS" and (pbs_distribution != None or pbs_parameters != None):
            raise ValueError(f"""You can only use 'pbs_distribution' and 'pbs_parameters' with 'd="PBS"'""")

    def run(self, simulation_time: float = float("inf")) -> pd.DataFrame:
        """Simulation of any queueing model.

        Args:
            simulation_time (float, optional): This variable defines the total duration of the simulation. It sets the length of time over which the simulation will model the system's behavior. Defaults to float("inf")
            number_of_simulations (int, optional): Number of simulations of the process. Can also be considered as the number of days or number of times you want to simulate your scenario. Defaults to 1.

        Returns:
            tuple: [description]
        """

        # Create a new variable with the simulation time
        self.__simulation_time = simulation_time

        # Simulation time has to be greater than 0
        if simulation_time <= 0:
            raise ValueError(f"""'simulation_time' has to be a number and cannot be less or equals than zero.""")

        # If it is infinity, create a massive number
        if simulation_time == float("inf"):
            # Big number for the actual probabilities
            simulation_time = round(self.__a.ppf(0.9999)) * 1000000

        # Results Dictionary - initialized with everything equals to 0
        simulation_results = dict()
        simulation_results["Attention Order"] = [0]
        simulation_results["Arrival Time"] = [0]
        simulation_results["Total Number of people"] = [0]
        simulation_results["Number of people in Line"] = [0]
        simulation_results["Time in Line"] = [0]
        simulation_results["Time in service"] = [0]
        simulation_results["Leave Time"] = [0]
        simulation_results["Join the system?"] = [0]
        # Servers Information - Depends on the number in self.__c
        for server in range(1, self.__c + 1):
            simulation_results[f"Time busy server {server}"] = [0]
        for server in range(1, self.__c + 1):
            simulation_results[f"Server {server} attended this element?"] = [0]

        # Selections according to the queue discipline
        if self.__d == "FIFO":
            simulation_results = self.__fifo(simulation_time, simulation_results)
        elif self.__d == "LIFO":
            simulation_results = self.__lifo(simulation_time, simulation_results)
        elif self.__d == "PBS":
            simulation_results = self.__pbs(simulation_time, simulation_results)

        # Create a new column that is the same for all queue disciplines, to determine if that person was attended after the "close time"
        simulation_results["Finish after closed"] = [1 if leave_time > self.__simulation_time else 0 for leave_time in simulation_results["Leave Time"]]

        # Convert result to a DataFrame
        self.__result_simulation = pd.DataFrame(simulation_results)
        self.__result_simulation = self.__result_simulation.drop(index=0).reset_index(drop=True)

        # Calculte probabilities for each number of elements
        self.elements_prob()

        return self.__result_simulation

    def __last_not_null(self, array: list) -> int:
        """Reviews all elements and returns the last one that is not null

        Args:
            array (list): Array that we need to identify the last element no null

        Returns:
            int: last element no null
        """
        for element in reversed(array):
            if not np.isnan(element):
                return element

    def __fifo(self, simulation_time: int, simulation_results: dict) -> dict:
        """Simulation of the FIFO queue discipline according to all input parameters

        Args:
            simulation_time (int): Max. Time the simulation would take
            simulation_results (dict): Dictionary where all answers will be stored

        Returns:
            dict: Returns simulation_results input variable with all the calculations
        """

        # Initialize some variable for this simulation
        arrivals = list()
        arriving_time = 0
        population = 0

        # Determine all the arrival hours if the number of people do not exceed the maximum or the arriving time is less than the Max. Simulation Time
        while arriving_time < simulation_time and population < self.__n:
            arr = self.__a.ppf(random.random())
            if arriving_time + arr < simulation_time:
                arrivals.append(arr)
                arriving_time += arrivals[-1]
                population += 1
            else:
                break

        # Start simulation for each arrival
        for arrival in arrivals:
            # Include that person time in the result
            simulation_results["Arrival Time"].append(simulation_results["Arrival Time"][-1] + arrival)

            # Number of people at that time
            number_of_people = 0
            start = simulation_results["Arrival Time"][-1]
            for other_person in range(len(simulation_results["Arrival Time"]) - 1):
                if simulation_results["Arrival Time"][other_person] <= start and simulation_results["Leave Time"][other_person] >= start:
                    number_of_people += 1
            # Plus one means that person in the system
            simulation_results["Total Number of people"].append(number_of_people + 1)

            # Calculte the number of people in line at that time
            if simulation_results["Total Number of people"][-1] <= self.__c:
                simulation_results["Number of people in Line"].append(0)
            else:
                simulation_results["Number of people in Line"].append(simulation_results["Total Number of people"][-1] - self.__c)

            # Verify if the number of people is less or equals the max value
            if simulation_results["Total Number of people"][-1] <= self.__k:

                # Join the system
                simulation_results["Join the system?"].append(1)

                # Attention order
                simulation_results["Attention Order"].append(max(simulation_results["Attention Order"]) + 1)

                # Review shortest time among all servers and choosing the first server that is available
                first_server_available = 0
                first_server_available_time = float("Inf")
                for server in range(1, self.__c + 1):
                    last_time_server_not_null = self.__last_not_null(simulation_results[f"Time busy server {server}"])
                    if last_time_server_not_null <= simulation_results["Arrival Time"][-1]:
                        first_server_available = server
                        first_server_available_time = last_time_server_not_null
                        break
                    elif last_time_server_not_null < first_server_available_time:
                        first_server_available = server
                        first_server_available_time = last_time_server_not_null

                # Assign the time in line if the arrival time is less the available server then he needed to do the line
                simulation_results["Time in Line"].append(
                    max(
                        first_server_available_time,
                        simulation_results["Arrival Time"][-1],
                    )
                    - simulation_results["Arrival Time"][-1]
                )
                # Simulate time in service
                simulation_results["Time in service"].append(self.__s.ppf(random.random()))
                # Leave time of that person
                simulation_results["Leave Time"].append(simulation_results["Arrival Time"][-1] + simulation_results["Time in Line"][-1] + simulation_results["Time in service"][-1])
                # Same as leave time is the max time busy server
                simulation_results[f"Time busy server {first_server_available}"].append(simulation_results["Leave Time"][-1])
                # This server was the enchanged of help the element
                simulation_results[f"Server {first_server_available} attended this element?"].append(1)

                # Keep same finish time to other servers and did not attend those servers
                for server in range(1, self.__c + 1):
                    if server != first_server_available:
                        simulation_results[f"Time busy server {server}"].append(simulation_results[f"Time busy server {server}"][-1])
                        simulation_results[f"Server {server} attended this element?"].append(0)
            else:
                # If the number of people is greater than the maximum allowed, then do not include any stat to that person
                simulation_results["Join the system?"].append(0)
                simulation_results["Attention Order"].append(np.nan)
                simulation_results["Time in Line"].append(np.nan)
                simulation_results["Time in service"].append(np.nan)
                simulation_results["Leave Time"].append(np.nan)
                # Keep same finish time to other servers and np.nan for all servers
                for server in range(1, self.__c + 1):
                    simulation_results[f"Time busy server {server}"].append(np.nan)
                    simulation_results[f"Server {server} attended this element?"].append(np.nan)

        return simulation_results

    def __lifo(self, simulation_time: int, simulation_results: dict) -> dict:
        """Simulation of the LIFO queue discipline according to all input parameters

        Args:
            simulation_time (int): Max. Time the simulation would take
            simulation_results (dict): Dictionary where all answers will be stored

        Returns:
            dict: Returns simulation_results input variable with all the calculations
        """

        # Initialize some variable for this simulation
        arrivals = list()
        arriving_time = 0
        population = 0

        # Dictionary to identify the order, we initialize it with the first row (value) that also is the first one attended (key) (everything it's zero)
        order_idx = {0: 0}

        # Determine all the arrival hours if the number of people do not exceed the maximum or the arriving time is less than the Max. Simulation Time
        while arriving_time < simulation_time and population < self.__n:
            arr = self.__a.ppf(random.random())
            if arriving_time + arr < simulation_time:
                arrivals.append(arr)
                arriving_time += arrivals[-1]
                population += 1
            else:
                break

        # Start simulation for each arrival
        for arrival in arrivals:
            # Review time of arrival
            simulation_results["Arrival Time"].append(simulation_results["Arrival Time"][-1] + arrival)

            # Last person that was served
            last_attended = max(simulation_results["Attention Order"])

            # If person that arrives time is greater than end of services of at least one machine, we can review people in line or this person to take the service, if not, go to the line
            go_to_queue = True
            for server in range(1, self.__c + 1):
                if simulation_results["Arrival Time"][-1] > simulation_results[f"Time busy server {server}"][order_idx[last_attended]]:
                    first_server_available = server
                    first_server_available_time = simulation_results[f"Time busy server {server}"][order_idx[last_attended]]
                    go_to_queue = False
                    break

            # If he needs to go to the line
            if go_to_queue == True:

                ## We should verify the number of people including him!!

                # Number of people at that time
                number_of_people = len(
                    list(
                        filter(
                            lambda x: True if x == -1 else False,
                            simulation_results["Attention Order"],
                        )
                    )
                )
                # Plus one means that person that has just arrived into the system
                simulation_results["Number of people in Line"].append(number_of_people + 1)
                # Total people
                simulation_results["Total Number of people"].append(simulation_results["Number of people in Line"][-1] + self.__c)

                # Can that person enter?
                if simulation_results["Total Number of people"][-1] <= self.__k:
                    # Add that person into the queue (send to queue = -1)
                    simulation_results["Join the system?"].append(1)
                    simulation_results["Attention Order"].append(-1)
                    simulation_results["Time in Line"].append(-1)
                    simulation_results["Time in service"].append(-1)
                    simulation_results["Leave Time"].append(-1)
                    # Keep same finish time to other servers
                    for server in range(1, self.__c + 1):
                        simulation_results[f"Time busy server {server}"].append(-1)
                        # This server was the enchanged of help the element
                        simulation_results[f"Server {server} attended this element?"].append(-1)
                # if not
                else:
                    # If the number of people is greater than the maximum allowed, then do not include any stat to that person
                    simulation_results["Join the system?"].append(0)
                    simulation_results["Attention Order"].append(np.nan)
                    simulation_results["Time in Line"].append(np.nan)
                    simulation_results["Time in service"].append(np.nan)
                    simulation_results["Leave Time"].append(np.nan)
                    # Keep same finish time to other servers
                    for server in range(1, self.__c + 1):
                        simulation_results[f"Time busy server {server}"].append(np.nan)
                        # This server was the enchanged of help the element
                        simulation_results[f"Server {server} attended this element?"].append(np.nan)

            else:

                ## We need to send the last element that arrived before him to the service, if there was nobody, we send the current person.
                # Number of people at that time
                number_of_people = len(
                    list(
                        filter(
                            lambda x: True if x == -1 else False,
                            simulation_results["Attention Order"],
                        )
                    )
                )
                # If there is nobody, we assign that element.
                if number_of_people == 0:
                    # Plus one means that person that has just arrived into the system
                    simulation_results["Number of people in Line"].append(number_of_people)
                    simulation_results["Join the system?"].append(1)

                    # Attention position
                    simulation_results["Attention Order"].append(max(simulation_results["Attention Order"]) + 1)

                    # Add the service time and additional information
                    simulation_results["Time in Line"].append(
                        max(
                            first_server_available_time,
                            simulation_results["Arrival Time"][-1],
                        )
                        - simulation_results["Arrival Time"][-1]
                    )
                    simulation_results["Time in service"].append(self.__s.ppf(random.random()))
                    simulation_results["Leave Time"].append(simulation_results["Arrival Time"][-1] + simulation_results["Time in Line"][-1] + simulation_results["Time in service"][-1])
                    simulation_results[f"Time busy server {first_server_available}"].append(simulation_results["Leave Time"][-1])
                    # This server was the enchanged of help the element
                    simulation_results[f"Server {first_server_available} attended this element?"].append(1)

                    # Keep same finish time to other servers
                    people_being_served = 0
                    for server in range(1, self.__c + 1):
                        if server != first_server_available:
                            simulation_results[f"Time busy server {server}"].append(max(simulation_results[f"Time busy server {server}"]))
                            # This server was the enchanged of help the element
                            simulation_results[f"Server {server} attended this element?"].append(0)
                            if simulation_results[f"Time busy server {server}"][-1] >= simulation_results["Arrival Time"][-1]:
                                people_being_served += 1
                        else:
                            people_being_served += 1

                    # Number of people at that time
                    simulation_results["Total Number of people"].append(people_being_served)
                    # Update order
                    order_idx[max(simulation_results["Attention Order"])] = len(simulation_results["Attention Order"]) - 1

                else:

                    # The people go from last to first, and each time a person passes, it must be checked if there are machines available before the person reaches this point. If not, the process stops, and this person is sent to the queue, and the process continues.

                    # Review from the end to the beggining without the element that has just arrived
                    for idx in range(len(simulation_results["Attention Order"]) - 1, -1, -1):
                        # If that element has not been served
                        if simulation_results["Attention Order"][idx] == -1:

                            min_time = float("Inf")
                            no_servers_available = True
                            # Search in all Servers which is available
                            for server in range(1, self.__c + 1):
                                # Review if servers are available (before last arrival (-1))
                                if (
                                    simulation_results["Arrival Time"][-1] > simulation_results[f"Time busy server {server}"][order_idx[last_attended]]
                                    and min_time > simulation_results[f"Time busy server {server}"][order_idx[last_attended]]
                                ):
                                    # Find Server number and time
                                    first_server_available = server
                                    first_server_available_time = simulation_results[f"Time busy server {server}"][order_idx[last_attended]]
                                    min_time = first_server_available_time
                                    # Let's review again
                                    no_servers_available = False

                            if no_servers_available == True:
                                break

                            else:

                                ## Assigning
                                # Attention position
                                simulation_results["Attention Order"][idx] = max(simulation_results["Attention Order"]) + 1

                                # Add the service time and additional information
                                simulation_results["Time in Line"][idx] = (
                                    max(
                                        first_server_available_time,
                                        simulation_results["Arrival Time"][idx],
                                    )
                                    - simulation_results["Arrival Time"][idx]
                                )
                                simulation_results["Time in service"][idx] = self.__s.ppf(random.random())
                                simulation_results["Leave Time"][idx] = simulation_results["Arrival Time"][idx] + simulation_results["Time in Line"][idx] + simulation_results["Time in service"][idx]
                                simulation_results[f"Time busy server {first_server_available}"][idx] = simulation_results["Leave Time"][idx]
                                # This server was the enchanged of help the element
                                simulation_results[f"Server {first_server_available} attended this element?"][idx] = 1

                                # Keep same finish time to other servers
                                for others_servers in range(1, self.__c + 1):
                                    if others_servers != first_server_available:
                                        simulation_results[f"Time busy server {others_servers}"][idx] = max(simulation_results[f"Time busy server {others_servers}"])
                                        # This server was the enchanged of help the element
                                        simulation_results[f"Server {others_servers} attended this element?"][idx] = 0

                                # Assign last attended as this one
                                last_attended = max(simulation_results["Attention Order"])
                                order_idx[last_attended] = idx

                    ### If there is no more servers available, we review if the person that has just arrived can join the system, if so we send it to the line
                    # New number of people after assignig to machines
                    number_of_people = len(
                        list(
                            filter(
                                lambda x: True if x == -1 else False,
                                simulation_results["Attention Order"],
                            )
                        )
                    )

                    ## Add to the queue
                    # Add last person into the queue
                    # Plus one means that person that has just arrived into the system
                    value_to_add = 1
                    people_being_served = 0
                    for server in range(1, self.__c + 1):
                        # Review if servers are available (before last arrival (-1)) we do not add a "1" in Number of people in line
                        if simulation_results["Arrival Time"][-1] > simulation_results[f"Time busy server {server}"][order_idx[last_attended]]:
                            value_to_add = 0
                        elif simulation_results[f"Time busy server {server}"][order_idx[last_attended]] >= simulation_results["Arrival Time"][-1]:
                            people_being_served += 1

                    ## After all send the last person to the queue (person that has just arrived)
                    # Assign all to queue
                    simulation_results["Number of people in Line"].append(number_of_people + value_to_add)
                    simulation_results["Total Number of people"].append(
                        simulation_results["Number of people in Line"][-1] + people_being_served + 1 if value_to_add == 0 else simulation_results["Number of people in Line"][-1] + people_being_served
                    )
                    simulation_results["Join the system?"].append(1)
                    simulation_results["Attention Order"].append(-1)
                    simulation_results["Time in Line"].append(-1)
                    simulation_results["Time in service"].append(-1)
                    simulation_results["Leave Time"].append(-1)
                    # Keep same finish time to other servers and servers have not attended this user
                    for server in range(1, self.__c + 1):
                        simulation_results[f"Time busy server {server}"].append(-1)
                        simulation_results[f"Server {server} attended this element?"].append(-1)

        ## Last people to assign
        # After "closing time" there are people that are not assign yet. This logic assign that people

        # Identify last person attended
        last_attended = max(simulation_results["Attention Order"])
        # Iterate from the last one to the first one if there is any element not assigned yet
        for idx in range(len(simulation_results["Attention Order"]) - 1, -1, -1):
            # if that element hasn't been attended yet
            if simulation_results["Attention Order"][idx] == -1:

                min_time = float("Inf")
                # Search in all Servers which is available
                for server in range(1, self.__c + 1):
                    # Review if servers are available (before last arrival (-1))
                    if min_time > simulation_results[f"Time busy server {server}"][order_idx[last_attended]]:
                        # Find Server number and time
                        min_time = simulation_results[f"Time busy server {server}"][order_idx[last_attended]]
                        first_server_available = server
                        first_server_available_time = simulation_results[f"Time busy server {server}"][order_idx[last_attended]]

                ## Assigning
                # Attention position
                simulation_results["Attention Order"][idx] = max(simulation_results["Attention Order"]) + 1

                # Add the service time and additional information
                simulation_results["Time in Line"][idx] = (
                    max(
                        first_server_available_time,
                        simulation_results["Arrival Time"][idx],
                    )
                    - simulation_results["Arrival Time"][idx]
                )
                simulation_results["Time in service"][idx] = self.__s.ppf(random.random())
                simulation_results["Leave Time"][idx] = simulation_results["Arrival Time"][idx] + simulation_results["Time in Line"][idx] + simulation_results["Time in service"][idx]
                simulation_results[f"Time busy server {first_server_available}"][idx] = simulation_results["Leave Time"][idx]
                simulation_results[f"Server {first_server_available} attended this element?"][idx] = 1

                # Keep same finish time to other servers
                for others_servers in range(1, self.__c + 1):
                    if others_servers != first_server_available:
                        simulation_results[f"Time busy server {others_servers}"][idx] = max(simulation_results[f"Time busy server {others_servers}"])
                        simulation_results[f"Server {others_servers} attended this element?"][idx] = 0

                # Assign last attended as this one
                last_attended = max(simulation_results["Attention Order"])
                order_idx[last_attended] = idx

        return simulation_results

    def __pbs(self, simulation_time: int, simulation_results: dict) -> dict:
        """Simulation of the PBS queue discipline according to all input parameters

        Args:
            simulation_time (int): Max. Time the simulation would take
            simulation_results (dict): Dictionary where all answers will be stored

        Returns:
            dict: Returns simulation_results input variable with all the calculations
        """
        # Initialize some variable for this simulation
        arrivals = list()
        all_priorities = list()
        arriving_time = 0
        population = 0
        # A new field for the result dictionary is created because of the PBS logic
        simulation_results["Priority"] = [0]

        # Dictionary to identify the order, we initialize it with the first row that also is the first one attended (everything it's zero)
        order_idx = {0: 0}

        # Determine all the arrival hours if the number of people do not exceed the maximum or the arriving time is less than the Max. Simulation Time. Also this line determines the priority of each element
        while arriving_time < simulation_time and population < self.__n:
            arr = self.__a.ppf(random.random())
            if arriving_time + arr < simulation_time:
                arrivals.append(arr)
                all_priorities.append(self.__label.ppf(random.random()))
                arriving_time += arrivals[-1]
                population += 1
            else:
                break

        # Start simulation for each arrival
        for index_arrival, arrival in enumerate(arrivals):
            simulation_results["Arrival Time"].append(simulation_results["Arrival Time"][-1] + arrival)
            simulation_results["Priority"].append(all_priorities[index_arrival])

            # Number of people at that time
            number_of_people = 0
            start = simulation_results["Arrival Time"][-1]

            # Number of people at that time
            for other_person in range(len(simulation_results["Arrival Time"]) - 1):
                if simulation_results["Arrival Time"][other_person] <= start and simulation_results["Leave Time"][other_person] >= start:
                    number_of_people += 1
                elif simulation_results["Attention Order"][other_person] == -1:
                    number_of_people += 1

            # Plus one means that person is in the system
            simulation_results["Total Number of people"].append(number_of_people + 1)

            # Determine the number of people in line at that time
            if simulation_results["Total Number of people"][-1] <= self.__c:
                simulation_results["Number of people in Line"].append(0)
            else:
                simulation_results["Number of people in Line"].append(simulation_results["Total Number of people"][-1] - self.__c)

            # Can that person enter?
            if simulation_results["Total Number of people"][-1] <= self.__k:

                # JOin the system
                simulation_results["Join the system?"].append(1)

                # Last person that was served
                last_attended = max(simulation_results["Attention Order"])

                # If person that arrives time is greater than end of services of at least one machine, we can review people in line or this person to take the service, if not, go to the line
                go_to_queue = True
                for server in range(1, self.__c + 1):
                    if simulation_results["Arrival Time"][-1] > simulation_results[f"Time busy server {server}"][order_idx[last_attended]]:
                        first_server_available = server
                        first_server_available_time = simulation_results[f"Time busy server {server}"][order_idx[last_attended]]
                        go_to_queue = False
                        break

                # If need to go to the queue
                if go_to_queue == True:

                    # Add that person into the queue (-1 means queue)

                    simulation_results["Attention Order"].append(-1)
                    simulation_results["Time in Line"].append(-1)
                    simulation_results["Time in service"].append(-1)
                    simulation_results["Leave Time"].append(-1)
                    # Keep same finish time to other servers
                    for server in range(1, self.__c + 1):
                        simulation_results[f"Time busy server {server}"].append(-1)
                        simulation_results[f"Server {server} attended this element?"].append(-1)

                else:

                    # IF there is nobody that person can be served
                    if simulation_results["Number of people in Line"][-1] == 0:

                        # Attention position
                        simulation_results["Attention Order"].append(max(simulation_results["Attention Order"]) + 1)

                        # Add the service time and additional information
                        simulation_results["Time in Line"].append(
                            max(
                                first_server_available_time,
                                simulation_results["Arrival Time"][-1],
                            )
                            - simulation_results["Arrival Time"][-1]
                        )
                        simulation_results["Time in service"].append(self.__s.ppf(random.random()))
                        simulation_results["Leave Time"].append(simulation_results["Arrival Time"][-1] + simulation_results["Time in Line"][-1] + simulation_results["Time in service"][-1])
                        simulation_results[f"Time busy server {first_server_available}"].append(simulation_results["Leave Time"][-1])
                        simulation_results[f"Server {first_server_available} attended this element?"].append(1)

                        # Keep same finish time to other servers
                        for server in range(1, self.__c + 1):
                            if server != first_server_available:
                                simulation_results[f"Time busy server {server}"].append(max(simulation_results[f"Time busy server {server}"]))
                                simulation_results[f"Server {server} attended this element?"].append(0)
                        # Update order list
                        order_idx[max(simulation_results["Attention Order"])] = len(simulation_results["Attention Order"]) - 1

                    else:

                        ## Add last person into the queue - This part helps to identify if the person needs to be attended, they could be attended first if priority is the highest
                        # Adding that person into the queue (just to verify priority)
                        simulation_results["Attention Order"].append(-1)
                        simulation_results["Time in Line"].append(-1)
                        simulation_results["Time in service"].append(-1)
                        simulation_results["Leave Time"].append(-1)
                        # Keep same finish time to other servers
                        for server in range(1, self.__c + 1):
                            simulation_results[f"Time busy server {server}"].append(-1)
                            simulation_results[f"Server {server} attended this element?"].append(-1)

                        # Bigger numbers are first priority, smaller numbers are less priority
                        priority_list = sorted(list(set(simulation_results["Priority"])), reverse=True)

                        # Verify priority (starts with bigger numbers)
                        for priority in priority_list:
                            for idx in range(len(simulation_results["Attention Order"])):
                                # Verify if person is in line and its priority
                                if simulation_results["Attention Order"][idx] == -1 and simulation_results["Priority"][idx] == priority:

                                    min_time = float("Inf")
                                    no_servers_available = True
                                    # Search in all Servers which is available
                                    for server in range(1, self.__c + 1):
                                        # Review if servers are available (before last arrival (-1))
                                        if (
                                            simulation_results["Arrival Time"][-1] > simulation_results[f"Time busy server {server}"][order_idx[last_attended]]
                                            and min_time > simulation_results[f"Time busy server {server}"][order_idx[last_attended]]
                                        ):
                                            # Find Server number and time
                                            first_server_available = server
                                            first_server_available_time = simulation_results[f"Time busy server {server}"][order_idx[last_attended]]
                                            min_time = first_server_available_time
                                            # Let's review again
                                            no_servers_available = False

                                    if no_servers_available == True:
                                        break

                                    else:

                                        ## Assigning
                                        # Attention position
                                        simulation_results["Attention Order"][idx] = max(simulation_results["Attention Order"]) + 1

                                        # Add the service time and additional information
                                        simulation_results["Time in Line"][idx] = (
                                            max(
                                                first_server_available_time,
                                                simulation_results["Arrival Time"][idx],
                                            )
                                            - simulation_results["Arrival Time"][idx]
                                        )
                                        simulation_results["Time in service"][idx] = self.__s.ppf(random.random())
                                        simulation_results["Leave Time"][idx] = (
                                            simulation_results["Arrival Time"][idx] + simulation_results["Time in Line"][idx] + simulation_results["Time in service"][idx]
                                        )
                                        simulation_results[f"Time busy server {first_server_available}"][idx] = simulation_results["Leave Time"][idx]
                                        simulation_results[f"Server {first_server_available} attended this element?"][idx] = 1

                                        # Keep same finish time to other servers
                                        for others_servers in range(1, self.__c + 1):
                                            if others_servers != first_server_available:
                                                simulation_results[f"Time busy server {others_servers}"][idx] = max(simulation_results[f"Time busy server {others_servers}"])
                                                simulation_results[f"Server {others_servers} attended this element?"][idx] = 0

                                        # Assign last attended as this one
                                        last_attended = max(simulation_results["Attention Order"])
                                        order_idx[last_attended] = idx

            # If not
            else:
                # If the number of people is greater than the maximum allowed, then do not include any stat to that person
                simulation_results["Join the system?"].append(0)
                simulation_results["Attention Order"].append(np.nan)
                simulation_results["Time in Line"].append(np.nan)
                simulation_results["Time in service"].append(np.nan)
                simulation_results["Leave Time"].append(np.nan)
                # Keep same finish time to other servers
                for server in range(1, self.__c + 1):
                    simulation_results[f"Time busy server {server}"].append(np.nan)
                    simulation_results[f"Server {server} attended this element?"].append(np.nan)

        ## Assign Missing elements
        # Bigger numbers are first priority, smaller numbers are less priority
        priority_list = sorted(list(set(simulation_results["Priority"])), reverse=True)
        # Verify priority (starts with bigger numbers)
        for priority in priority_list:
            # Verify if that element has not been attended and if belongs to the actual priority
            for idx in range(len(simulation_results["Attention Order"])):
                if simulation_results["Attention Order"][idx] == -1 and simulation_results["Priority"][idx] == priority:

                    min_time = float("Inf")
                    # Search in all Servers which is available
                    for server in range(1, self.__c + 1):
                        # Review if servers are available (before last arrival (-1))
                        if min_time > simulation_results[f"Time busy server {server}"][order_idx[last_attended]]:
                            # Find Server number and time
                            first_server_available = server
                            first_server_available_time = simulation_results[f"Time busy server {server}"][order_idx[last_attended]]
                            min_time = first_server_available_time

                    ## Assigning
                    # Attention position
                    simulation_results["Attention Order"][idx] = max(simulation_results["Attention Order"]) + 1

                    # Add the service time and additional information
                    simulation_results["Time in Line"][idx] = (
                        max(
                            first_server_available_time,
                            simulation_results["Arrival Time"][idx],
                        )
                        - simulation_results["Arrival Time"][idx]
                    )
                    simulation_results["Time in service"][idx] = self.__s.ppf(random.random())
                    simulation_results["Leave Time"][idx] = simulation_results["Arrival Time"][idx] + simulation_results["Time in Line"][idx] + simulation_results["Time in service"][idx]
                    simulation_results[f"Time busy server {first_server_available}"][idx] = simulation_results["Leave Time"][idx]
                    simulation_results[f"Server {first_server_available} attended this element?"][idx] = 1

                    # Keep same finish time to other servers
                    for others_servers in range(1, self.__c + 1):
                        if others_servers != first_server_available:
                            simulation_results[f"Time busy server {others_servers}"][idx] = max(simulation_results[f"Time busy server {others_servers}"])
                            simulation_results[f"Server {others_servers} attended this element?"][idx] = 0

                    # Assign last attended as this one
                    last_attended = max(simulation_results["Attention Order"])
                    order_idx[last_attended] = idx

        return simulation_results

    def to_csv(self, file_name: str, index: bool = True) -> None:
        """Simulation results to CVS

        Args:
            file_name (str): File Name to add to the CSV file. You should include ".csv" at the end of your file
            index (bool, optional): Defaults to True. Add index in CSV file.
        """
        if len(self.__result_simulation) == 0:
            raise ValueError(f"""You need to run the simulation to use this""")
        else:
            self.__result_simulation.to_csv(file_name, index=index)

    def to_excel(self, file_name: str, sheet_name: str = "Sheet1", index: bool = True) -> None:
        """Simulation results to Excel File

        Args:
            file_name (str): File Name to add to the Excel file. You should include ".xlsx" at the end of your file
            index (bool, optional): Defaults to True. Add index in Excel file.
        """
        if len(self.__result_simulation) == 0:
            raise ValueError(f"""You need to run the simulation to use this""")
        else:
            self.__result_simulation.to_excel(file_name, index=index, sheet_name=sheet_name)

    def system_utilization(self) -> float:
        """Returns system utilization according to simulation

        Returns:
            float: System Utilization
        """
        return self.__result_simulation["Time in service"].sum() / self.__simulation_time

    def no_clients_prob(self) -> float:
        """Probability of no having clients

        Returns:
            float: No clients probability
        """
        return 1 - self.__result_simulation["Time in service"].sum() / self.__simulation_time

    def elements_prob(self, bins: int = 50000) -> dict:
        """Creates the probability for each number of elements. Example: Probability to be 0, prob. to be 1, prob. to be 2... depending on simulation values

        Args:
            bins (int, optional): Number of intervals to determine the probability to be in each stage. Defaults to 50000.

        Returns:
            dict: Element and probability result
        """
        multiplier = 1
        step = 0
        while step < 1:
            # Range to determine probabilities
            max_val = int(self.__result_simulation["Leave Time"].max() * multiplier)
            min_val = int(self.__result_simulation["Arrival Time"].min() * multiplier)
            step = int((max_val - min_val) / bins)
            if step < 1:
                multiplier = multiplier * 10

        # Definimos un rango de tiempos para analizar la cantidad de clientes en el sistema
        time_points = [round(t, 2) for t in range(min_val, max_val, step)]

        # Number of clients in system in each instant
        customers_at_time = [((self.__result_simulation["Arrival Time"] <= t / 100) & (self.__result_simulation["Leave Time"] >= t / 100)).sum() for t in time_points]

        # Count a number of times each number appears
        count_customers = collections.Counter(customers_at_time)

        # Calculate probability per number of customer
        total_points = len(time_points)
        self.number_probabilities = {k: v / total_points for k, v in count_customers.items()}

        return self.number_probabilities

    def number_elements_prob(self, number: int, prob_type: str) -> float:
        """Calculates the probability Exact, less or equals or greater or equals.

        Args:
            number (int): Number that we want to identify the different probabilities
            prob_type (str): Could be one of the following options: 'exact_value', 'greater_equals', 'less_equals'

        Returns:
            float: Probability of the number of elements
        """
        if isinstance(number, int) == False:
            raise ValueError(f"""Number can only be integer""")

        if prob_type == "exact_value":
            return self.number_probabilities[number]
        elif prob_type == "greater_equals":
            return sum([self.number_probabilities[key] for key in self.number_probabilities.keys() if key >= number])
        elif prob_type == "less_equals":
            return sum([self.number_probabilities[key] for key in self.number_probabilities.keys() if key <= number])
        else:
            raise ValueError(f"""You can only select one of the following prob_type: 'exact_value', 'greater_equals', 'less_equals'""")

    def average_time_system(self) -> float:
        """Average time in system

        Returns:
            float: Average time in system
        """
        return (self.__result_simulation["Time in service"] + self.__result_simulation["Time in Line"]).mean()

    def average_time_queue(self) -> float:
        """Average time in queue

        Returns:
            float: Average time in queue
        """
        return self.__result_simulation["Time in Line"].mean()

    def average_time_service(self) -> float:
        """Average time in service

        Returns:
            float: Average time in service
        """
        return self.__result_simulation["Time in service"].mean()

    def standard_deviation_time_system(self) -> float:
        """Standard Deviation time in system

        Returns:
            float: Standard Deviation time in system
        """
        return (self.__result_simulation["Time in service"] + self.__result_simulation["Time in Line"]).std()

    def standard_deviation_time_queue(self) -> float:
        """Standard Deviation time in queue

        Returns:
            float: Standard Deviation time in queue
        """
        return self.__result_simulation["Time in Line"].std()

    def standard_deviation_time_service(self) -> float:
        """Standard Deviation time in service

        Returns:
            float: Standard Deviation time in service
        """
        return self.__result_simulation["Time in service"].std()

    def average_elements_system(self) -> float:
        """Average elements in system

        Returns:
            float: Average elements in system
        """
        return (self.__result_simulation["Time in service"] + self.__result_simulation["Time in Line"]).sum() / self.__simulation_time

    def average_elements_queue(self) -> float:
        """Average elements in queue

        Returns:
            float: Average elements in queue
        """
        return (self.__result_simulation["Time in Line"]).sum() / self.__simulation_time

    def probability_to_join_system(self) -> float:
        """Probability to join the system

        Returns:
            float: Probability to join the system
        """
        return (self.__result_simulation["Join the system?"]).sum() / len(self.__result_simulation)

    def probability_to_finish_after_time(self) -> float:
        """Probability to finish after time

        Returns:
            float: Probability to finish after time
        """
        return (self.__result_simulation["Finish after closed"]).sum() / len(self.__result_simulation)

    def probability_to_wait_in_line(self) -> float:
        """Probability to wait in the queue

        Returns:
            float: Probability to wait in the queue
        """
        result = np.where(self.__result_simulation["Time in Line"] > 0, 1, 0)

        return result.sum() / len(self.__result_simulation)

    def servers_utilization(self) -> pd.DataFrame:
        """Determine the server utilization according to the simulation result

        Returns:
            pd.DataFrame: Utilization of all servers, you can find the server number in the rows
        """
        # Calculate server utilization
        serv_util_dict = {
            f"Utilization Server #{server}": self.__result_simulation[f"Server {server} attended this element?"].sum() / len(self.__result_simulation) for server in range(1, self.__c + 1)
        }
        # Convert into a DataFrame
        df = pd.DataFrame.from_dict(serv_util_dict, orient="index").rename(columns={0: "Value"})

        df.index.name = "Metrics"

        return df.reset_index()

    def number_probability_summary(self) -> pd.DataFrame:
        """Returns the probability for each element. The probability is Exact, less or equals or greater or equals; represented in each column.

        Returns:
            pd.DataFrame: Dataframe with all the needed probabilities for each element.
        """

        options = ["less_equals", "exact_value", "greater_equals"]

        dictionaty_number = {key: [self.number_elements_prob(int(key), option) for option in options] for key in self.number_probabilities.keys()}

        df = pd.DataFrame.from_dict(dictionaty_number, orient="index").rename(
            columns={
                0: "Prob. Less or Equals",
                1: "Exact Probability",
                2: "Prob. Greter or equals",
            }
        )

        df.index.name = "Number of elements"

        return df.reset_index()

    def metrics_summary(self) -> pd.DataFrame:
        """Returns the summary of the following metrics: Average Time in System, Average Time in Queue, Average Time in Service, Std. Dev. Time in System, Std. Dev. Time in Queue, Std. Dev. Time in Service, Average Elements in System, Average Elements in Queue, Probability to join the System, Probability to finish after Time, Probability to Wait in Line

        Returns:
            pd.DataFrame: Returns dataframe with all the information
        """
        metrics = dict()
        metrics["Average Time in System"] = float(self.average_time_system())
        metrics["Average Time in Queue"] = float(self.average_time_queue())
        metrics["Average Time in Service"] = float(self.average_time_service())
        metrics["Std. Dev. Time in System"] = float(self.standard_deviation_time_system())
        metrics["Std. Dev. Time in Queue"] = float(self.standard_deviation_time_queue())
        metrics["Std. Dev. Time in Service"] = float(self.standard_deviation_time_service())
        metrics["Average Elements in System"] = float(self.average_elements_system())
        metrics["Average Elements in Queue"] = float(self.average_elements_queue())
        metrics["Probability to join the System"] = float(self.probability_to_join_system())
        metrics["Probability to finish after Time"] = float(self.probability_to_finish_after_time())
        metrics["Probability to Wait in Line"] = float(self.probability_to_wait_in_line())

        df = pd.DataFrame.from_dict(metrics, orient="index").rename(columns={0: "Value"})

        df.index.name = "Metrics"

        df = df.reset_index()

        df_2 = self.servers_utilization()

        df = pd.concat([df, df_2], axis=0)

        return df.reset_index(drop=True)

    def confidence_interval_metrics(
        self,
        simulation_time: int = float("Inf"),
        confidence_level: int = 0.95,
        replications: int = 30,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate a confidence interval for probabilities and metrics.

        Args:
            simulation_time (int, optional): Simulation time. Defaults to float("Inf)
            confidence_level (int, optional): Confidence level for the confidence interval for all the metrics and probabilities. Defaults to 0.95.
            replications (int, optional): Number of samples of simulations to create. Defaults to 30.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Returns probabilities and metrics dataframe with confidene interval for all metrics.
        """
        # Initializa variables
        tot_prob = pd.DataFrame()
        tot_metrics = pd.DataFrame()
        # Run replications
        for _ in range(replications):
            # Initialize simulation to avoid issues
            self.__init__(
                self.__save_a,
                self.__save_a_params,
                self.__save_s,
                self.__save_s_params,
                self.__c,
                self.__k,
                self.__n,
                self.__d,
                self.__pbs_distribution,
                self.__pbs_parameters,
            )
            # Run simulation
            self.run(simulation_time)
            # Save metrics and probabilities
            number_probability_summary = self.number_probability_summary()
            metrics_summary = self.metrics_summary()
            # Concat previous results with current results
            tot_prob = pd.concat([tot_prob, number_probability_summary])
            tot_metrics = pd.concat([tot_metrics, metrics_summary])

        # First Confidence Interval
        std__ = tot_prob.groupby(["Number of elements"]).std()
        mean__ = tot_prob.groupby(["Number of elements"]).mean()

        standard_error = std__ / math.sqrt(replications)
        normal_standard = phitter.continuous.Normal({"mu": 0, "sigma": 1})
        z = normal_standard.ppf((1 + confidence_level) / 2)
        ## Confidence Interval
        avg = mean__.copy()
        lower_bound = (
            (mean__ - (z * standard_error))
            .copy()
            .rename(
                columns={
                    "Prob. Less or Equals": "LB - Prob. Less or Equals",
                    "Exact Probability": "LB - Exact Probability",
                    "Prob. Greter or equals": "LB - Prob. Greater or equals",
                }
            )
        )
        upper_bound = (
            (mean__ + (z * standard_error))
            .copy()
            .rename(
                columns={
                    "Prob. Less or Equals": "UB - Prob. Less or Equals",
                    "Exact Probability": "UB - Exact Probability",
                    "Prob. Greter or equals": "UB - Prob. Greater or equals",
                }
            )
        )
        avg = avg.rename(
            columns={
                "Prob. Less or Equals": "AVG - Prob. Less or Equals",
                "Exact Probability": "AVG - Exact Probability",
                "Prob. Greter or equals": "AVG - Prob. Greater or equals",
            }
        )
        tot_prob_interval = pd.concat([lower_bound, avg, upper_bound], axis=1)
        tot_prob_interval = tot_prob_interval[
            [
                "LB - Prob. Less or Equals",
                "AVG - Prob. Less or Equals",
                "UB - Prob. Less or Equals",
                "LB - Exact Probability",
                "AVG - Exact Probability",
                "UB - Exact Probability",
                "LB - Prob. Greater or equals",
                "AVG - Prob. Greater or equals",
                "UB - Prob. Greater or equals",
            ]
        ]

        # Second Confidence Interval
        std__2 = tot_metrics.groupby(["Metrics"]).std()
        mean__2 = tot_metrics.groupby(["Metrics"]).mean()

        standard_error = std__2 / math.sqrt(replications)
        normal_standard = phitter.continuous.Normal({"mu": 0, "sigma": 1})
        z = normal_standard.ppf((1 + confidence_level) / 2)
        ## Confidence Interval
        avg__2 = mean__2.copy()
        lower_bound = (mean__2 - (z * standard_error)).copy().rename(columns={"Value": "LB - Value"})
        upper_bound = (mean__2 + (z * standard_error)).copy().rename(columns={"Value": "UB - Value"})
        avg__2 = avg__2.rename(columns={"Value": "AVG - Value"})
        tot_metrics_interval = pd.concat([lower_bound, avg__2, upper_bound], axis=1)
        tot_metrics_interval = tot_metrics_interval[["LB - Value", "AVG - Value", "UB - Value"]]

        return tot_prob_interval.reset_index(), tot_metrics_interval.reset_index()
