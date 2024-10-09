import phitter
import pandas as pd
import random
import numpy as np


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
            d (str, optional): Queue discipline. This describes the rule or policy that determines the order in which customers are served. Common disciplines include First-In-First-Out ("FIFO"), Last-In-First-Out ("LIFO"), priority-based service ("PBS"). The queue discipline impacts waiting times and the overall fairness of the system.. Defaults to "FIFO".
            simulation_time (float, optional): This variable defines the total duration of the simulation. It sets the length of time over which the simulation will model the system's behavior. Defaults to float("inf")
            number_of_simulations (int, optional): Number of simulations of the process. Can also be considered as the number of days or number of times you want to simulate your scenario. Defaults to 1.
            pbs_distribution (str | None, optional): Discrete distribution that identifies the label of the pbs, this parameter can only be used with "d='PBS'". Distributions that can be used: 'own_distribution', 'bernoulli', 'binomial', 'geometric', 'hypergeometric', 'logarithmic', 'negative_binomial', 'poisson'. Defaults to None.
            pbs_parameters (dict | None, optional): Parameters of the discrete distribution that identifies the label of the pbs, this parameter can only be used with "d='PBS'". If it is 'own-distribution' add labels in the following way (example): {0: 0.5, 1: 0.3, 2: 0.2}. Where the "key" corresponds to the label and the "value" the probability whose total sum must add up to 1; "keys" with greater importances are the smallers and always have to be numeric keys. You can add as labels as you need.
        """

        self.__probability_distribution = (
            phitter.continuous.CONTINUOUS_DISTRIBUTIONS
            | phitter.discrete.DISCRETE_DISTRIBUTIONS
        )

        self.__pbs_distribution = {
            "own_distribution": OwnDistributions
        } | phitter.discrete.DISCRETE_DISTRIBUTIONS

        self.__queue_discipline = ["FIFO", "LIFO", "PBS"]

        self.__verify_variables(a, s, c, k, n, d, pbs_distribution, pbs_parameters)

        self.__a = self.__probability_distribution[a](a_parameters)
        self.__s = self.__probability_distribution[s](s_parameters)
        self.__c = c
        self.__k = k
        self.__n = n
        self.__d = d

        if d == "PBS":
            self.__label = self.__pbs_distribution[pbs_distribution](pbs_parameters)

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
            pbs_distribution (str): Label Distribution.
            pbs_parameters (dict): Parameters of the PBS Distribution
        """

        # Verify if a and s belong to a actual probability distribution
        if (
            a not in self.__probability_distribution.keys()
            or s not in self.__probability_distribution.keys()
        ):
            raise ValueError(
                f"""Please select one of the following probability distributions: '{"', '".join(self.__probability_distribution.keys())}'."""
            )
        # Verify Number of Servers
        if c <= 0:
            raise ValueError(
                f"""'c' has to be a number and cannot be less or equals than zero."""
            )
        # Verify Maximum System Capacity
        if k <= 0:
            raise ValueError(
                f"""'k' has to be a number and cannot be less or equals than zero."""
            )
        # Verify Total population of potential customers.
        if n <= 0:
            raise ValueError(
                f"""'n' has to be a number and cannot be less or equals than zero."""
            )

        if d not in self.__queue_discipline:
            raise ValueError(
                f"""'d' has to be one of the following queue discipline: '{"', '".join(self.__queue_discipline)}'."""
            )

        if k < c:
            raise ValueError(f"""'k' cannot be less than the number of servers (c)""")

        if d == "PBS":
            if pbs_distribution != None and pbs_parameters != None:
                if pbs_distribution not in self.__pbs_distribution:
                    raise ValueError(
                        f"""You should select one of the following distributions: {self.__pbs_distribution}"""
                    )
            elif pbs_distribution == None and pbs_parameters == None:
                raise ValueError(
                    f"""You must include 'pbs_distribution' and 'pbs_parameters' if you want to use 'PBS'."""
                )
        elif d != "PBS" and (pbs_distribution != None or pbs_parameters != None):
            raise ValueError(
                f"""You can only use 'pbs_distribution' and 'pbs_parameters' with 'd="PBS"'"""
            )

    def run(
        self, simulation_time: float = float("inf"), number_of_simulations: int = 1
    ) -> tuple:
        """Simulation of any queueing model.

        Args:
            simulation_time (float, optional): This variable defines the total duration of the simulation. It sets the length of time over which the simulation will model the system's behavior. Defaults to float("inf")
            number_of_simulations (int, optional): Number of simulations of the process. Can also be considered as the number of days or number of times you want to simulate your scenario. Defaults to 1.

        Returns:
            tuple: [description]
        """
        if simulation_time <= 0:
            raise ValueError(
                f"""'simulation_time' has to be a number and cannot be less or equals than zero."""
            )

        if number_of_simulations <= 0:
            raise ValueError(
                f"""'number_of_simulations' has to be a number and cannot be less or equals than zero."""
            )

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
        # Servers Information
        for server in range(1, self.__c + 1):
            simulation_results[f"Time busy server {server}"] = [0]

        if self.__d == "FIFO":
            simulation_results = self.__fifo(simulation_time, simulation_results)
        elif self.__d == "LIFO":
            simulation_results = self.__lifo(simulation_time, simulation_results)
        elif self.__d == "PBS":
            simulation_results = self.__pbs(simulation_time, simulation_results)

        return simulation_results

    def __last_not_null(self, array):
        for element in reversed(array):
            if not np.isnan(element):
                return element

    def __fifo(self, simulation_time, simulation_results):
        arrivals = list()
        arriving_time = 0
        population = 0
        # Determine all the arrival hours
        while arriving_time < simulation_time and population < self.__n:
            arrivals.append(self.__a.ppf(random.random()))
            arriving_time += arrivals[-1]
            population += 1

        for arrival in arrivals:
            simulation_results["Arrival Time"].append(
                simulation_results["Arrival Time"][-1] + arrival
            )

            # Number of people at that time
            number_of_people = 0
            start = simulation_results["Arrival Time"][-1]
            for other_person in range(len(simulation_results["Arrival Time"]) - 1):
                if (
                    simulation_results["Arrival Time"][other_person] <= start
                    and simulation_results["Leave Time"][other_person] >= start
                ):
                    number_of_people += 1
            # Plus one means that person in the system
            simulation_results["Total Number of people"].append(number_of_people + 1)

            if simulation_results["Total Number of people"][-1] <= self.__c:
                simulation_results["Number of people in Line"].append(0)
            else:
                simulation_results["Number of people in Line"].append(
                    simulation_results["Total Number of people"][-1] - self.__c
                )

            if simulation_results["Total Number of people"][-1] <= self.__k:

                # JOin the system
                simulation_results["Join the system?"].append(1)

                # Attention order
                simulation_results["Attention Order"].append(
                    max(simulation_results["Attention Order"]) + 1
                )

                # Review shortest time among all servers and choosing the first server that is available
                first_server_available = 0
                first_server_available_time = float("Inf")
                for server in range(1, self.__c + 1):
                    last_time_server_not_null = self.__last_not_null(
                        simulation_results[f"Time busy server {server}"]
                    )
                    if (
                        last_time_server_not_null
                        <= simulation_results["Arrival Time"][-1]
                    ):
                        first_server_available = server
                        first_server_available_time = last_time_server_not_null
                        break
                    elif last_time_server_not_null < first_server_available_time:
                        first_server_available = server
                        first_server_available_time = last_time_server_not_null

                simulation_results["Time in Line"].append(
                    max(
                        first_server_available_time,
                        simulation_results["Arrival Time"][-1],
                    )
                    - simulation_results["Arrival Time"][-1]
                )
                simulation_results["Time in service"].append(
                    self.__s.ppf(random.random())
                )
                simulation_results["Leave Time"].append(
                    simulation_results["Arrival Time"][-1]
                    + simulation_results["Time in Line"][-1]
                    + simulation_results["Time in service"][-1]
                )
                simulation_results[f"Time busy server {first_server_available}"].append(
                    simulation_results["Leave Time"][-1]
                )

                # Keep same finish time to other servers
                for server in range(1, self.__c + 1):
                    if server != first_server_available:
                        simulation_results[f"Time busy server {server}"].append(
                            simulation_results[f"Time busy server {server}"][-1]
                        )
            else:
                simulation_results["Join the system?"].append(0)
                simulation_results["Attention Order"].append(np.nan)
                simulation_results["Time in Line"].append(np.nan)
                simulation_results["Time in service"].append(np.nan)
                simulation_results["Leave Time"].append(np.nan)
                # Keep same finish time to other servers
                for server in range(1, self.__c + 1):
                    simulation_results[f"Time busy server {server}"].append(np.nan)

        return simulation_results

    def __lifo(self, simulation_time, simulation_results):
        arrivals = list()
        arriving_time = 0
        population = 0

        # Dictionary to identify the order, we initialize it with the first row that also is the first one attended (everything it's zero)
        order_idx = {0: 0}

        # Determine all the arrival hours
        while arriving_time < simulation_time and population < self.__n:
            arrivals.append(self.__a.ppf(random.random()))
            arriving_time += arrivals[-1]
            population += 1

        for arrival in arrivals:
            # Review time of arrival
            simulation_results["Arrival Time"].append(
                simulation_results["Arrival Time"][-1] + arrival
            )

            # Last person that was served
            last_attended = max(simulation_results["Attention Order"])

            # If person that arrives time is greater than end of services of at least one machine, we can review people in line or this person to take the service, if not, go to the line
            go_to_queue = True
            for server in range(1, self.__c + 1):
                if (
                    simulation_results["Arrival Time"][-1]
                    > simulation_results[f"Time busy server {server}"][
                        order_idx[last_attended]
                    ]
                ):
                    first_server_available = server
                    first_server_available_time = simulation_results[
                        f"Time busy server {server}"
                    ][order_idx[last_attended]]
                    go_to_queue = False
                    break

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
                simulation_results["Number of people in Line"].append(
                    number_of_people + 1
                )
                # Total people
                simulation_results["Total Number of people"].append(
                    simulation_results["Number of people in Line"][-1] + self.__c
                )

                # Can that person enter?
                if simulation_results["Total Number of people"][-1] <= self.__k:
                    # Add that person into the queue

                    simulation_results["Join the system?"].append(1)
                    simulation_results["Attention Order"].append(-1)
                    simulation_results["Time in Line"].append(-1)
                    simulation_results["Time in service"].append(-1)
                    simulation_results["Leave Time"].append(-1)
                    # Keep same finish time to other servers
                    for server in range(1, self.__c + 1):
                        simulation_results[f"Time busy server {server}"].append(-1)
                # if not
                else:
                    simulation_results["Join the system?"].append(0)
                    simulation_results["Attention Order"].append(np.nan)
                    simulation_results["Time in Line"].append(np.nan)
                    simulation_results["Time in service"].append(np.nan)
                    simulation_results["Leave Time"].append(np.nan)
                    # Keep same finish time to other servers
                    for server in range(1, self.__c + 1):
                        simulation_results[f"Time busy server {server}"].append(np.nan)

            else:

                ## We need to send the last element that arrived before him to the service, if there was nobody, we send the current person

                number_of_people = len(
                    list(
                        filter(
                            lambda x: True if x == -1 else False,
                            simulation_results["Attention Order"],
                        )
                    )
                )

                if number_of_people == 0:
                    # Plus one means that person that has just arrived into the system
                    simulation_results["Number of people in Line"].append(
                        number_of_people
                    )
                    simulation_results["Join the system?"].append(1)

                    # Attention position
                    simulation_results["Attention Order"].append(
                        max(simulation_results["Attention Order"]) + 1
                    )

                    # Add the service time and additional information
                    simulation_results["Time in Line"].append(
                        max(
                            first_server_available_time,
                            simulation_results["Arrival Time"][-1],
                        )
                        - simulation_results["Arrival Time"][-1]
                    )
                    simulation_results["Time in service"].append(
                        self.__s.ppf(random.random())
                    )
                    simulation_results["Leave Time"].append(
                        simulation_results["Arrival Time"][-1]
                        + simulation_results["Time in Line"][-1]
                        + simulation_results["Time in service"][-1]
                    )
                    simulation_results[
                        f"Time busy server {first_server_available}"
                    ].append(simulation_results["Leave Time"][-1])

                    # Keep same finish time to other servers
                    people_being_served = 0
                    for server in range(1, self.__c + 1):
                        if server != first_server_available:
                            simulation_results[f"Time busy server {server}"].append(
                                max(simulation_results[f"Time busy server {server}"])
                            )
                            if (
                                simulation_results[f"Time busy server {server}"][-1]
                                >= simulation_results["Arrival Time"][-1]
                            ):
                                people_being_served += 1
                        else:
                            people_being_served += 1

                    simulation_results["Total Number of people"].append(
                        people_being_served
                    )

                    order_idx[max(simulation_results["Attention Order"])] = (
                        len(simulation_results["Attention Order"]) - 1
                    )

                else:

                    # pendiente logica de varias personas y varias maquinas para asignar, la personas de los ultimos a primeros y cada que pase una persona se debe ver si hay maquinas disponibles antes de que llegue la persona de este punto, ahi muere y se manda a esta persona a la fila y se continua el proceso

                    for idx in range(
                        len(simulation_results["Attention Order"]) - 1, -1, -1
                    ):
                        if simulation_results["Attention Order"][idx] == -1:

                            min_time = float("Inf")
                            no_servers_available = True
                            # Search in all Servers which is available
                            for server in range(1, self.__c + 1):
                                # Review if servers are available (before last arrival (-1))
                                if (
                                    simulation_results["Arrival Time"][-1]
                                    > simulation_results[f"Time busy server {server}"][
                                        order_idx[last_attended]
                                    ]
                                    and min_time
                                    > simulation_results[f"Time busy server {server}"][
                                        order_idx[last_attended]
                                    ]
                                ):
                                    # Find Server number and time
                                    first_server_available = server
                                    first_server_available_time = simulation_results[
                                        f"Time busy server {server}"
                                    ][order_idx[last_attended]]
                                    min_time = first_server_available_time
                                    # Let's review again
                                    no_servers_available = False

                            if no_servers_available == True:
                                break

                            else:

                                ## Assigning
                                # Attention position
                                simulation_results["Attention Order"][idx] = (
                                    max(simulation_results["Attention Order"]) + 1
                                )

                                # Add the service time and additional information
                                simulation_results["Time in Line"][idx] = (
                                    max(
                                        first_server_available_time,
                                        simulation_results["Arrival Time"][idx],
                                    )
                                    - simulation_results["Arrival Time"][idx]
                                )
                                simulation_results["Time in service"][idx] = (
                                    self.__s.ppf(random.random())
                                )
                                simulation_results["Leave Time"][idx] = (
                                    simulation_results["Arrival Time"][idx]
                                    + simulation_results["Time in Line"][idx]
                                    + simulation_results["Time in service"][idx]
                                )
                                simulation_results[
                                    f"Time busy server {first_server_available}"
                                ][idx] = simulation_results["Leave Time"][idx]

                                # Keep same finish time to other servers
                                for others_servers in range(1, self.__c + 1):
                                    if others_servers != first_server_available:
                                        simulation_results[
                                            f"Time busy server {others_servers}"
                                        ][idx] = max(
                                            simulation_results[
                                                f"Time busy server {others_servers}"
                                            ]
                                        )

                                # Assign last attended as this one
                                last_attended = max(
                                    simulation_results["Attention Order"]
                                )
                                order_idx[last_attended] = idx

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
                        if (
                            simulation_results["Arrival Time"][-1]
                            > simulation_results[f"Time busy server {server}"][
                                order_idx[last_attended]
                            ]
                        ):
                            value_to_add = 0
                        elif (
                            simulation_results[f"Time busy server {server}"][
                                order_idx[last_attended]
                            ]
                            >= simulation_results["Arrival Time"][-1]
                        ):
                            people_being_served += 1

                    simulation_results["Number of people in Line"].append(
                        number_of_people + value_to_add
                    )
                    simulation_results["Total Number of people"].append(
                        simulation_results["Number of people in Line"][-1]
                        + people_being_served
                        + 1
                        if value_to_add == 0
                        else simulation_results["Number of people in Line"][-1]
                        + people_being_served
                    )
                    simulation_results["Join the system?"].append(1)
                    simulation_results["Attention Order"].append(-1)
                    simulation_results["Time in Line"].append(-1)
                    simulation_results["Time in service"].append(-1)
                    simulation_results["Leave Time"].append(-1)
                    # Keep same finish time to other servers
                    for server in range(1, self.__c + 1):
                        simulation_results[f"Time busy server {server}"].append(-1)

        ## Last people to assign

        last_attended = max(simulation_results["Attention Order"])
        for idx in range(len(simulation_results["Attention Order"]) - 1, -1, -1):
            if simulation_results["Attention Order"][idx] == -1:

                min_time = float("Inf")
                # Search in all Servers which is available
                for server in range(1, self.__c + 1):
                    # Review if servers are available (before last arrival (-1))
                    if (
                        min_time
                        > simulation_results[f"Time busy server {server}"][
                            order_idx[last_attended]
                        ]
                    ):
                        # Find Server number and time
                        min_time = simulation_results[f"Time busy server {server}"][
                            order_idx[last_attended]
                        ]
                        first_server_available = server
                        first_server_available_time = simulation_results[
                            f"Time busy server {server}"
                        ][order_idx[last_attended]]

                ## Assigning
                # Attention position
                simulation_results["Attention Order"][idx] = (
                    max(simulation_results["Attention Order"]) + 1
                )

                # Add the service time and additional information
                simulation_results["Time in Line"][idx] = (
                    max(
                        first_server_available_time,
                        simulation_results["Arrival Time"][idx],
                    )
                    - simulation_results["Arrival Time"][idx]
                )
                simulation_results["Time in service"][idx] = self.__s.ppf(
                    random.random()
                )
                simulation_results["Leave Time"][idx] = (
                    simulation_results["Arrival Time"][idx]
                    + simulation_results["Time in Line"][idx]
                    + simulation_results["Time in service"][idx]
                )
                simulation_results[f"Time busy server {first_server_available}"][
                    idx
                ] = simulation_results["Leave Time"][idx]

                # Keep same finish time to other servers
                for others_servers in range(1, self.__c + 1):
                    if others_servers != first_server_available:
                        simulation_results[f"Time busy server {others_servers}"][
                            idx
                        ] = max(
                            simulation_results[f"Time busy server {others_servers}"]
                        )

                # Assign last attended as this one
                last_attended = max(simulation_results["Attention Order"])
                order_idx[last_attended] = idx

        return simulation_results

    def __pbs(self, simulation_time, simulation_results):

        arrivals = list()
        all_priorities = list()
        arriving_time = 0
        population = 0
        simulation_results["Priority"] = [0]

        # Dictionary to identify the order, we initialize it with the first row that also is the first one attended (everything it's zero)
        order_idx = {0: 0}

        # Determine all the arrival hours
        while arriving_time < simulation_time and population < self.__n:
            arrivals.append(self.__a.ppf(random.random()))
            all_priorities.append(self.__label.ppf(random.random()))
            arriving_time += arrivals[-1]
            population += 1

        for index_arrival, arrival in enumerate(arrivals):
            simulation_results["Arrival Time"].append(
                simulation_results["Arrival Time"][-1] + arrival
            )
            simulation_results["Priority"].append(all_priorities[index_arrival])

            # Number of people at that time
            number_of_people = 0
            start = simulation_results["Arrival Time"][-1]

            for other_person in range(len(simulation_results["Arrival Time"]) - 1):
                if (
                    simulation_results["Arrival Time"][other_person] <= start
                    and simulation_results["Leave Time"][other_person] >= start
                ):
                    number_of_people += 1
                elif simulation_results["Attention Order"][other_person] == -1:
                    number_of_people += 1

            # Plus one means that person in the system
            simulation_results["Total Number of people"].append(number_of_people + 1)

            if simulation_results["Total Number of people"][-1] <= self.__c:
                simulation_results["Number of people in Line"].append(0)
            else:
                simulation_results["Number of people in Line"].append(
                    simulation_results["Total Number of people"][-1] - self.__c
                )

            if simulation_results["Total Number of people"][-1] <= self.__k:

                # JOin the system
                simulation_results["Join the system?"].append(1)

                # Last person that was served
                last_attended = max(simulation_results["Attention Order"])

                # If person that arrives time is greater than end of services of at least one machine, we can review people in line or this person to take the service, if not, go to the line
                go_to_queue = True
                for server in range(1, self.__c + 1):
                    if (
                        simulation_results["Arrival Time"][-1]
                        > simulation_results[f"Time busy server {server}"][
                            order_idx[last_attended]
                        ]
                    ):
                        first_server_available = server
                        first_server_available_time = simulation_results[
                            f"Time busy server {server}"
                        ][order_idx[last_attended]]
                        go_to_queue = False
                        break

                if go_to_queue == True:

                    # Add that person into the queue

                    simulation_results["Attention Order"].append(-1)
                    simulation_results["Time in Line"].append(-1)
                    simulation_results["Time in service"].append(-1)
                    simulation_results["Leave Time"].append(-1)
                    # Keep same finish time to other servers
                    for server in range(1, self.__c + 1):
                        simulation_results[f"Time busy server {server}"].append(-1)

                else:

                    if simulation_results["Number of people in Line"][-1] == 0:

                        # Attention position
                        simulation_results["Attention Order"].append(
                            max(simulation_results["Attention Order"]) + 1
                        )

                        # Add the service time and additional information
                        simulation_results["Time in Line"].append(
                            max(
                                first_server_available_time,
                                simulation_results["Arrival Time"][-1],
                            )
                            - simulation_results["Arrival Time"][-1]
                        )
                        simulation_results["Time in service"].append(
                            self.__s.ppf(random.random())
                        )
                        simulation_results["Leave Time"].append(
                            simulation_results["Arrival Time"][-1]
                            + simulation_results["Time in Line"][-1]
                            + simulation_results["Time in service"][-1]
                        )
                        simulation_results[
                            f"Time busy server {first_server_available}"
                        ].append(simulation_results["Leave Time"][-1])

                        # Keep same finish time to other servers
                        for server in range(1, self.__c + 1):
                            if server != first_server_available:
                                simulation_results[f"Time busy server {server}"].append(
                                    max(
                                        simulation_results[f"Time busy server {server}"]
                                    )
                                )

                        order_idx[max(simulation_results["Attention Order"])] = (
                            len(simulation_results["Attention Order"]) - 1
                        )

                    else:

                        # Add last person into the queue - This part helps to identify if the person needs to be attended, they coulb be attended first if priority is the highest

                        simulation_results["Attention Order"].append(-1)
                        simulation_results["Time in Line"].append(-1)
                        simulation_results["Time in service"].append(-1)
                        simulation_results["Leave Time"].append(-1)
                        # Keep same finish time to other servers
                        for server in range(1, self.__c + 1):
                            simulation_results[f"Time busy server {server}"].append(-1)

                        # Bigger numbers are first priority, smaller numbers are less priority
                        priority_list = sorted(
                            list(set(simulation_results["Priority"])), reverse=True
                        )

                        for priority in priority_list:
                            for idx in range(
                                len(simulation_results["Attention Order"])
                            ):
                                if (
                                    simulation_results["Attention Order"][idx] == -1
                                    and simulation_results["Priority"][idx] == priority
                                ):

                                    min_time = float("Inf")
                                    no_servers_available = True
                                    # Search in all Servers which is available
                                    for server in range(1, self.__c + 1):
                                        # Review if servers are available (before last arrival (-1))
                                        if (
                                            simulation_results["Arrival Time"][-1]
                                            > simulation_results[
                                                f"Time busy server {server}"
                                            ][order_idx[last_attended]]
                                            and min_time
                                            > simulation_results[
                                                f"Time busy server {server}"
                                            ][order_idx[last_attended]]
                                        ):
                                            # Find Server number and time
                                            first_server_available = server
                                            first_server_available_time = (
                                                simulation_results[
                                                    f"Time busy server {server}"
                                                ][order_idx[last_attended]]
                                            )
                                            min_time = first_server_available_time
                                            # Let's review again
                                            no_servers_available = False

                                    if no_servers_available == True:
                                        break

                                    else:

                                        ## Assigning
                                        # Attention position
                                        simulation_results["Attention Order"][idx] = (
                                            max(simulation_results["Attention Order"])
                                            + 1
                                        )

                                        # Add the service time and additional information
                                        simulation_results["Time in Line"][idx] = (
                                            max(
                                                first_server_available_time,
                                                simulation_results["Arrival Time"][idx],
                                            )
                                            - simulation_results["Arrival Time"][idx]
                                        )
                                        simulation_results["Time in service"][idx] = (
                                            self.__s.ppf(random.random())
                                        )
                                        simulation_results["Leave Time"][idx] = (
                                            simulation_results["Arrival Time"][idx]
                                            + simulation_results["Time in Line"][idx]
                                            + simulation_results["Time in service"][idx]
                                        )
                                        simulation_results[
                                            f"Time busy server {first_server_available}"
                                        ][idx] = simulation_results["Leave Time"][idx]

                                        # Keep same finish time to other servers
                                        for others_servers in range(1, self.__c + 1):
                                            if others_servers != first_server_available:
                                                simulation_results[
                                                    f"Time busy server {others_servers}"
                                                ][idx] = max(
                                                    simulation_results[
                                                        f"Time busy server {others_servers}"
                                                    ]
                                                )

                                        # Assign last attended as this one
                                        last_attended = max(
                                            simulation_results["Attention Order"]
                                        )
                                        order_idx[last_attended] = idx

            else:
                simulation_results["Join the system?"].append(0)
                simulation_results["Attention Order"].append(np.nan)
                simulation_results["Time in Line"].append(np.nan)
                simulation_results["Time in service"].append(np.nan)
                simulation_results["Leave Time"].append(np.nan)
                # Keep same finish time to other servers
                for server in range(1, self.__c + 1):
                    simulation_results[f"Time busy server {server}"].append(np.nan)

        ## Assign Missing elements
        # Bigger numbers are first priority, smaller numbers are less priority
        priority_list = sorted(list(set(simulation_results["Priority"])), reverse=True)

        for priority in priority_list:
            for idx in range(len(simulation_results["Attention Order"])):
                if (
                    simulation_results["Attention Order"][idx] == -1
                    and simulation_results["Priority"][idx] == priority
                ):

                    min_time = float("Inf")
                    # Search in all Servers which is available
                    for server in range(1, self.__c + 1):
                        # Review if servers are available (before last arrival (-1))
                        if (
                            min_time
                            > simulation_results[f"Time busy server {server}"][
                                order_idx[last_attended]
                            ]
                        ):
                            # Find Server number and time
                            first_server_available = server
                            first_server_available_time = simulation_results[
                                f"Time busy server {server}"
                            ][order_idx[last_attended]]
                            min_time = first_server_available_time

                    ## Assigning
                    # Attention position
                    simulation_results["Attention Order"][idx] = (
                        max(simulation_results["Attention Order"]) + 1
                    )

                    # Add the service time and additional information
                    simulation_results["Time in Line"][idx] = (
                        max(
                            first_server_available_time,
                            simulation_results["Arrival Time"][idx],
                        )
                        - simulation_results["Arrival Time"][idx]
                    )
                    simulation_results["Time in service"][idx] = self.__s.ppf(
                        random.random()
                    )
                    simulation_results["Leave Time"][idx] = (
                        simulation_results["Arrival Time"][idx]
                        + simulation_results["Time in Line"][idx]
                        + simulation_results["Time in service"][idx]
                    )
                    simulation_results[f"Time busy server {first_server_available}"][
                        idx
                    ] = simulation_results["Leave Time"][idx]

                    # Keep same finish time to other servers
                    for others_servers in range(1, self.__c + 1):
                        if others_servers != first_server_available:
                            simulation_results[f"Time busy server {others_servers}"][
                                idx
                            ] = max(
                                simulation_results[f"Time busy server {others_servers}"]
                            )

                    # Assign last attended as this one
                    last_attended = max(simulation_results["Attention Order"])
                    order_idx[last_attended] = idx

        return simulation_results
