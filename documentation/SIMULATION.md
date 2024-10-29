# Simulation

## Process Simulation

This will help you to understand your processes. To use it, run the following line

```python
from phitter import simulation

# Create a simulation process instance
simulation = simulation.ProcessSimulation()

```

### Add processes to your simulation instance

There are two ways to add processes to your simulation instance:

-   Adding a **process _without_ preceding process (new branch)**
-   Adding a **process _with_ preceding process (with previous ids)**

#### Process _without_ preceding process (new branch)

```python
# Add a new process without preceding process
simulation.add_process(
    prob_distribution="normal",
    parameters={"mu": 5, "sigma": 2},
    process_id="first_process",
    number_of_products=10,
    number_of_servers=3,
    new_branch=True,
)

```

#### Process _with_ preceding process (with previous ids)

```python
# Add a new process with preceding process
simulation.add_process(
    prob_distribution="exponential",
    parameters={"lambda": 4},
    process_id="second_process",
    previous_ids=["first_process"],
)

```

#### All together and adding some new process

The order in which you add each process **_matters_**. You can add as many processes as you need.

```python
# Add a new process without preceding process
simulation.add_process(
    prob_distribution="normal",
    parameters={"mu": 5, "sigma": 2},
    process_id="first_process",
    number_of_products=10,
    number_of_servers=3,
    new_branch=True,
)

# Add a new process with preceding process
simulation.add_process(
    prob_distribution="exponential",
    parameters={"lambda": 4},
    process_id="second_process",
    previous_ids=["first_process"],
)

# Add a new process with preceding process
simulation.add_process(
    prob_distribution="gamma",
    parameters={"alpha": 15, "beta": 3},
    process_id="third_process",
    previous_ids=["first_process"],
)

# Add a new process without preceding process
simulation.add_process(
    prob_distribution="exponential",
    parameters={"lambda": 4.3},
    process_id="fourth_process",
    new_branch=True,
)


# Add a new process with preceding process
simulation.add_process(
    prob_distribution="beta",
    parameters={"alpha": 1, "beta": 1, "A": 2, "B": 3},
    process_id="fifth_process",
    previous_ids=["second_process", "fourth_process"],
)

# Add a new process with preceding process
simulation.add_process(
    prob_distribution="normal",
    parameters={"mu": 15, "sigma": 2},
    process_id="sixth_process",
    previous_ids=["third_process", "fifth_process"],
)
```

### Visualize your processes

You can visualize your processes to see if what you're trying to simulate is your actual process.

```python
# Graph your process
simulation.process_graph()
```

![Simulation](./multimedia/simulation_process_graph.png)

### Start Simulation

You can simulate and have different simulation time values or you can create a confidence interval for your process

#### Run Simulation

Simulate several scenarios of your complete process

```python
# Run Simulation
simulation.run(number_of_simulations=100)

# After run
simulation: pandas.Dataframe
```

### Review Simulation Metrics by Stage

If you want to review average time and standard deviation by stage run this line of code

```python
# Review simulation metrics
simulation.simulation_metrics() -> pandas.Dataframe
```

#### Run confidence interval

If you want to have a confidence interval for the simulation metrics, run the following line of code

```python
# Confidence interval for Simulation metrics
simulation.run_confidence_interval(
    confidence_level=0.99,
    number_of_simulations=100,
    replications=10,
) -> pandas.Dataframe
```

## Queue Simulation

If you need to simulate queues run the following code:

```python
from phitter import simulation

# Create a simulation process instance
simulation = simulation.QueueingSimulation(
    a="exponential",
    a_paramters={"lambda": 5},
    s="exponential",
    s_parameters={"lambda": 20},
    c=3,
)
```

In this case we are going to simulate **a** (arrivals) with _exponential distribution_ and **s** (service) as _exponential distribution_ with **c** equals to 3 different servers.

By default Maximum Capacity **k** is _infinity_, total population **n** is _infinity_ and the queue discipline **d** is _FIFO_. As we are not selecting **d** equals to "PBS" we don't have any information to add for **pbs_distribution** nor **pbs_parameters**

### Run the simulation

If you want to have the simulation results

```python
# Run simulation
simulation = simulation.run(simulation_time = 2000)
simulation: pandas.Dataframe
```

If you want to see some metrics and probabilities from this simulation you should use::

```python
# Calculate metrics
simulation.metrics_summary() -> pandas.Dataframe

# Calculate probabilities
number_probability_summary() -> pandas.Dataframe
```

### Run Confidence Interval for metrics and probabilities

If you want to have a confidence interval for your metrics and probabilities you should run the following line

```python
# Calculate confidence interval for metrics and probabilities
probabilities, metrics = simulation.confidence_interval_metrics(
    simulation_time=2000,
    confidence_level=0.99,
    replications=10,
)

probabilities -> pandas.Dataframe
metrics -> pandas.Dataframe
```
