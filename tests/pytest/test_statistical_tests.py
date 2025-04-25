import pytest

import phitter


def get_data(path: str) -> list[float | int]:
    sample_distribution_file = open(path, "r", encoding="utf-8-sig")
    data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
    sample_distribution_file.close()
    return data


@pytest.mark.parametrize("id_distribution", phitter.continuous.CONTINUOUS_DISTRIBUTIONS.keys())
def test_distribution_fit(id_distribution):
    distribution_class = phitter.continuous.CONTINUOUS_DISTRIBUTIONS[id_distribution]
    data = get_data(f"./distributions_samples/continuous_distributions_sample/sample_{id_distribution}.txt")
    continuous_measures = phitter.continuous.ContinuousMeasures(data)
    distribution = distribution_class(continuous_measures=continuous_measures)

    test_chi2 = phitter.continuous.evaluate_continuous_test_chi_square(distribution, continuous_measures)
    test_ks = phitter.continuous.evaluate_continuous_test_kolmogorov_smirnov(distribution, continuous_measures)
    test_ad = phitter.continuous.evaluate_continuous_test_anderson_darling(distribution, continuous_measures)

    n_test_passed = 3 - int(test_chi2["rejected"]) - int(test_ks["rejected"]) - int(test_ad["rejected"])

    assert n_test_passed > 0, f"La distribución {id_distribution} no pasó ninguna prueba"
