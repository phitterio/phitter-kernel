import pytest
import phitter
import pandas as pd


# Test that adding a process with a wrong function name raises a ValueError
def test_wrong_simulation_name():
    simulation = phitter.simulation.ProcessSimulation()
    with pytest.raises(ValueError):
        simulation.add_process(
            prob_distribution="non_real_function",
            parameters={"error": 13},
            process_id="first",
            new_branch=True,
            number_of_products=10,
        )


# Test to ensure the simulation runs correctly and returns a list of the correct length
def test_simulation_running():
    simulation = phitter.simulation.ProcessSimulation()
    simulation.add_process(
        prob_distribution="normal",
        parameters={"mu": 5, "sigma": 2},
        process_id="first",
        new_branch=True,
        number_of_products=10,
    )
    simulation.add_process(
        prob_distribution="exponential",
        parameters={"lambda": 4},
        process_id="second",
        previous_ids=["first"],
    )
    simulation.add_process(
        prob_distribution="exponential",
        parameters={"lambda": 4},
        process_id="ni_idea",
        previous_ids=["first"],
    )
    simulation.add_process(
        prob_distribution="exponential",
        parameters={"lambda": 4},
        process_id="ni_idea_2",
        previous_ids=["first"],
        number_of_products=15,
        number_of_servers=4,
    )

    simulation.add_process(
        prob_distribution="gamma",
        parameters={"alpha": 15, "beta": 3},
        process_id="third",
        new_branch=True,
    )
    # simulation.add_process("exponential", {"lambda": 4.3}, "nn", previous_ids=["third"])
    simulation.add_process(
        prob_distribution="beta",
        parameters={"alpha": 1, "beta": 1, "A": 2, "B": 3},
        process_id="fourth",
        previous_ids=["second", "third"],
        number_of_products=2,
    )
    simulation.add_process(
        prob_distribution="exponential",
        parameters={"lambda": 4.3},
        process_id="nn",
        previous_ids=["third"],
    )

    simulation.add_process(
        prob_distribution="exponential",
        parameters={"lambda": 4.3},
        process_id="fifth",
        new_branch=True,
    )
    simulation.add_process(
        prob_distribution="exponential",
        parameters={"lambda": 4.3},
        process_id="sixth",
        previous_ids=["fourth", "ni_idea", "nn"],
    )
    simulation.add_process(
        prob_distribution="beta",
        parameters={"alpha": 1, "beta": 1, "A": 2, "B": 3},
        process_id="seventh",
        previous_ids=["fifth", "sixth", "ni_idea_2"],
    )

    result = simulation.run(100)

    assert isinstance(result, pd.DataFrame)
    assert len(result["Total Simulation Time"]) == 100
    assert len(result.columns) == 11


# Test to check if the confidence interval results are as expected
def test_simulation_confidence_interval():
    simulation = phitter.simulation.ProcessSimulation()
    simulation = phitter.simulation.ProcessSimulation()
    simulation.add_process(
        prob_distribution="normal",
        parameters={"mu": 5, "sigma": 2},
        process_id="first",
        new_branch=True,
        number_of_products=10,
    )
    simulation.add_process(
        prob_distribution="exponential",
        parameters={"lambda": 4},
        process_id="second",
        previous_ids=["first"],
    )
    simulation.add_process(
        prob_distribution="exponential",
        parameters={"lambda": 4},
        process_id="ni_idea",
        previous_ids=["first"],
    )
    simulation.add_process(
        prob_distribution="exponential",
        parameters={"lambda": 4},
        process_id="ni_idea_2",
        previous_ids=["first"],
        number_of_products=15,
        number_of_servers=4,
    )

    simulation.add_process(
        prob_distribution="gamma",
        parameters={"alpha": 15, "beta": 3},
        process_id="third",
        new_branch=True,
    )
    # simulation.add_process("exponential", {"lambda": 4.3}, "nn", previous_ids=["third"])
    simulation.add_process(
        prob_distribution="beta",
        parameters={"alpha": 1, "beta": 1, "A": 2, "B": 3},
        process_id="fourth",
        previous_ids=["second", "third"],
        number_of_products=2,
    )
    simulation.add_process(
        prob_distribution="exponential",
        parameters={"lambda": 4.3},
        process_id="nn",
        previous_ids=["third"],
    )

    simulation.add_process(
        prob_distribution="exponential",
        parameters={"lambda": 4.3},
        process_id="fifth",
        new_branch=True,
    )
    simulation.add_process(
        prob_distribution="exponential",
        parameters={"lambda": 4.3},
        process_id="sixth",
        previous_ids=["fourth", "ni_idea", "nn"],
    )
    simulation.add_process(
        prob_distribution="beta",
        parameters={"alpha": 1, "beta": 1, "A": 2, "B": 3},
        process_id="seventh",
        previous_ids=["fifth", "sixth", "ni_idea_2"],
    )

    result = simulation.run_confidence_interval(number_of_simulations=100)

    assert len(result.columns) == 4
    assert isinstance(result, pd.DataFrame)
