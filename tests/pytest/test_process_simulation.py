import pytest
import phitter


def wrong_simulation_name():
    simulation = phitter.simulation.ProcessSimulation()
    with pytest.raises(ValueError):
        simulation.add_process(
            "non_real_function",
            {"error": 13},
            "first",
            new_branch=True,
            number_of_products=10,
        )


def simulation_running_assert():
    simulation = phitter.simulation.ProcessSimulation()
    simulation.add_process(
        "normal", {"mu": 5, "sigma": 2}, "first", new_branch=True, number_of_products=10
    )
    simulation.add_process(
        "exponential", {"lambda": 4}, "second", previous_ids=["first"]
    )
    simulation.add_process(
        "exponential", {"lambda": 4}, "ni_idea", previous_ids=["first"]
    )
    simulation.add_process(
        "exponential", {"lambda": 4}, "ni_idea_2", previous_ids=["first"]
    )

    simulation.add_process("gamma", {"alpha": 15, "beta": 3}, "third", new_branch=True)
    # simulation.add_process("exponential", {"lambda": 4.3}, "nn", previous_ids=["third"])
    simulation.add_process(
        "beta",
        {"alpha": 1, "beta": 1, "A": 2, "B": 3},
        "fourth",
        previous_ids=["second", "third"],
        number_of_products=2,
    )
    simulation.add_process("exponential", {"lambda": 4.3}, "nn", previous_ids=["third"])

    simulation.add_process("exponential", {"lambda": 4.3}, "fifth", new_branch=True)
    simulation.add_process(
        "exponential",
        {"lambda": 4.3},
        "sixth",
        previous_ids=["fourth", "ni_idea", "nn"],
    )
    simulation.add_process(
        "beta",
        {"alpha": 1, "beta": 1, "A": 2, "B": 3},
        "seventh",
        previous_ids=["fifth", "sixth", "ni_idea_2"],
    )

    result = simulation.run(10)

    assert len(result) == 10 and type(result) == list


def simulation_confidence_interval_assert():
    simulation = phitter.simulation.ProcessSimulation()
    simulation.add_process(
        "normal", {"mu": 5, "sigma": 2}, "first", new_branch=True, number_of_products=10
    )
    simulation.add_process(
        "exponential", {"lambda": 4}, "second", previous_ids=["first"]
    )
    simulation.add_process(
        "exponential", {"lambda": 4}, "ni_idea", previous_ids=["first"]
    )
    simulation.add_process(
        "exponential", {"lambda": 4}, "ni_idea_2", previous_ids=["first"]
    )

    simulation.add_process("gamma", {"alpha": 15, "beta": 3}, "third", new_branch=True)
    # simulation.add_process("exponential", {"lambda": 4.3}, "nn", previous_ids=["third"])
    simulation.add_process(
        "beta",
        {"alpha": 1, "beta": 1, "A": 2, "B": 3},
        "fourth",
        previous_ids=["second", "third"],
        number_of_products=2,
    )
    simulation.add_process("exponential", {"lambda": 4.3}, "nn", previous_ids=["third"])

    simulation.add_process("exponential", {"lambda": 4.3}, "fifth", new_branch=True)
    simulation.add_process(
        "exponential",
        {"lambda": 4.3},
        "sixth",
        previous_ids=["fourth", "ni_idea", "nn"],
    )
    simulation.add_process(
        "beta",
        {"alpha": 1, "beta": 1, "A": 2, "B": 3},
        "seventh",
        previous_ids=["fifth", "sixth", "ni_idea_2"],
    )

    result = simulation.run_confidence_interval(number_of_simulations=100)

    assert len(result) == 4 and type(result) == tuple
