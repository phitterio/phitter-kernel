import pandas
import phitter


def test_own_distribution_and_pbs():
    parameters = {0: 0.5, 1: 0.3, 2: 0.2}
    simulation = phitter.simulation.QueueingSimulation(
        "exponential",
        {"lambda": 5},
        "exponential",
        {"lambda": 20},
        3,
        d="PBS",
        pbs_distribution="own_distribution",
        pbs_parameters=parameters,
    )
    simulation.run(2000)
    assert simulation.probability_to_finish_after_time() <= 1


def test_confidence_interval_fifo():
    simulation = phitter.simulation.QueueingSimulation(
        "exponential", {"lambda": 5}, "exponential", {"lambda": 20}, 3, d="FIFO"
    )
    a, b = simulation.confidence_interval_metrics(2000, replications=10)
    assert (
        type(a) == pandas.DataFrame
        and len(a) > 0
        and type(b) == pandas.DataFrame
        and len(b) > 0
    )


def test_lifo_metrics():
    simulation = phitter.simulation.QueueingSimulation(
        "exponential",
        {"lambda": 5},
        "exponential",
        {"lambda": 20},
        3,
        n=100,
        k=3,
        d="LIFO",
    )
    simulation.run(2000)
    metrics = simulation.metrics_summary()
    prob = simulation.number_probability_summary()
    assert len(metrics) > 0 and len(prob) > 0
