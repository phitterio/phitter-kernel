import pytest
import pandas as pd
import phitter


# Own Simulation and distribution PBS
def test_own_distribution_and_pbs():
    parameters = {0: 0.5, 1: 0.3, 2: 0.2}
    simulation = phitter.simulation.QueueingSimulation(
        a="exponential",
        a_parameters={"lambda": 5},
        s="exponential",
        s_parameters={"lambda": 20},
        c=3,
        d="PBS",
        pbs_distribution="own_distribution",
        pbs_parameters=parameters,
    )
    # Run simulation with 2000 iterations
    simulation.run(simulation_time=2000)
    # Verify if the probability to finish after time is less than or equal to 1
    assert simulation.probability_to_finish_after_time() <= 1


# FIFO test for correct metrics and probabilities
def test_confidence_interval_fifo():
    simulation = phitter.simulation.QueueingSimulation(
        a="exponential",
        a_parameters={"lambda": 5},
        s="exponential",
        s_parameters={"lambda": 20},
        c=3,
        d="FIFO",
    )
    # Confidene interval with 10 replications
    a, b = simulation.confidence_interval_metrics(simulation_time=2000, replications=10)

    # Verify that both results are dataframes and not empty
    assert isinstance(a, pd.DataFrame) and len(a) > 0
    assert isinstance(b, pd.DataFrame) and len(b) > 0


# Test to verify if LIFO generates correct metrics and probabilities
def test_lifo_metrics():
    simulation = phitter.simulation.QueueingSimulation(
        a="exponential",
        a_parameters={"lambda": 5},
        s="exponential",
        s_parameters={"lambda": 20},
        c=3,
        n=100,
        k=3,
        d="LIFO",
    )
    # Simulation with 2000 simulations
    simulation.run(simulation_time=2000)

    # Obtain metrics and probabilities
    metrics = simulation.metrics_summary()
    prob = simulation.number_probability_summary()

    # Verify that this is not null
    assert len(metrics) > 0
    assert len(prob) > 0
