import numpy
import pytest

import phitter


def get_data(path: str) -> list[float | int]:
    sample_distribution_file = open(path, "r", encoding="utf-8-sig")
    data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
    sample_distribution_file.close()
    return data


@pytest.mark.parametrize("id_distribution", phitter.continuous.CONTINUOUS_DISTRIBUTIONS.keys())
def test_distribution_methods(id_distribution):
    distribution_class = phitter.continuous.CONTINUOUS_DISTRIBUTIONS[id_distribution]
    data = get_data(f"./phitter/continuous/continuous_distributions_sample/sample_{id_distribution}.txt")
    continuous_measures = phitter.continuous.ContinuousMeasures(data)
    distribution = distribution_class(continuous_measures=continuous_measures)

    try:
        # Basic information
        assert isinstance(distribution.name, str)
        assert isinstance(distribution.parameters, dict)

        # Test CDF
        assert isinstance(distribution.cdf(continuous_measures.mean), float)
        assert isinstance(distribution.cdf(numpy.array([continuous_measures.mean, continuous_measures.mean])), numpy.ndarray)

        # Test PDF
        assert isinstance(distribution.pdf(continuous_measures.mean), float)
        assert isinstance(distribution.pdf(numpy.array([continuous_measures.mean, continuous_measures.mean])), numpy.ndarray)

        # Test PPF
        ppf_value = distribution.ppf(0.5)
        assert isinstance(ppf_value, float)
        assert isinstance(distribution.ppf(numpy.array([0.5, 0.5])), numpy.ndarray)

        # Test sampling
        assert len(distribution.sample(5)) == 5

        # # Test statistics
        assert isinstance(distribution.mean, (float, int, type(None)))
        assert isinstance(distribution.variance, (float, int, type(None)))
        assert isinstance(distribution.skewness, (float, int, type(None)))
        assert isinstance(distribution.kurtosis, (float, int, type(None)))
        assert isinstance(distribution.median, (float, int, type(None)))
        assert isinstance(distribution.mode, (float, int, type(None)))

    except Exception as e:
        pytest.fail(f"Error en la distribución {id_distribution}: {str(e)}")
