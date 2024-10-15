import pytest
import pandas as pd
import phitter


# Test para simulación con distribución propia y política PBS
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
    # Ejecutar la simulación con 2000 iteraciones
    simulation.run(2000)
    # Verificar que la probabilidad de terminar después de un cierto tiempo es <= 1
    assert simulation.probability_to_finish_after_time() <= 1


# Test para verificar que la política FIFO genera intervalos de confianza correctos
def test_confidence_interval_fifo():
    simulation = phitter.simulation.QueueingSimulation(
        "exponential", {"lambda": 5}, "exponential", {"lambda": 20}, 3, d="FIFO"
    )
    # Obtener los intervalos de confianza para las métricas con 10 réplicas
    a, b = simulation.confidence_interval_metrics(2000, replications=10)

    # Verificar que ambos resultados son DataFrames de pandas y no están vacíos
    assert isinstance(a, pd.DataFrame) and len(a) > 0
    assert isinstance(b, pd.DataFrame) and len(b) > 0


# Test para verificar que la política LIFO genera métricas y probabilidades correctas
def test_lifo_metrics():
    simulation = phitter.simulation.QueueingSimulation(
        "exponential",
        {"lambda": 5},
        "exponential",
        {"lambda": 20},
        3,
        n=100,  # Número de eventos
        k=3,  # Capacidad del sistema
        d="LIFO",  # Política LIFO
    )
    # Ejecutar la simulación con 2000 iteraciones
    simulation.run(2000)

    # Obtener el resumen de métricas y la probabilidad numérica
    metrics = simulation.metrics_summary()
    prob = simulation.number_probability_summary()

    # Verificar que las métricas y probabilidades no están vacías
    assert len(metrics) > 0
    assert len(prob) > 0
