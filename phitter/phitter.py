import sys

sys.path.append("../continuous")
sys.path.append("../continuous/measurements")
sys.path.append("./continuous")
sys.path.append("../discrete")
sys.path.append("../discrete/measurements")
sys.path.append("./discrete")


from continuous.phitter_continuous import PHITTER_CONTINUOUS
from discrete.phitter_discrete import PHITTER_DISCRETE


class PHITTER:
    def __init__(
        self,
        data: list[int | float],
        fit_type="continuous",
        num_bins: int | None = None,
        confidence_level=0.95,
        minimum_sse=float("inf"),
    ):
        self.data = data
        self.fit_type = fit_type
        self.sorted_results_sse = None
        self.not_rejected_results = None
        self.num_bins = num_bins
        self.confidence_level = confidence_level
        self.minimum_sse = minimum_sse

    def fit(self, n_jobs: int = 1):
        if self.fit_type == "continuous":
            phitter_continuous = PHITTER_CONTINUOUS(
                self.data,
                self.num_bins,
                self.confidence_level,
                self.minimum_sse,
            )
            self.sorted_results_sse, self.not_rejected_results = phitter_continuous.fit(n_jobs=n_jobs)

        if self.fit_type == "discrete":
            phitter_discrete = PHITTER_DISCRETE(
                self.data,
                self.confidence_level,
                self.minimum_sse,
            )
            self.sorted_results_sse, self.not_rejected_results = phitter_discrete.fit(n_jobs=n_jobs)


if __name__ == "__main__":
    path = "../data/discrete/book2.txt"
    sample_distribution_file = open(path, "r", encoding="utf-8-sig")
    # for x in sample_distribution_file.read().splitlines():
    #     print(x)
    #     print(float(x.strip().replace(",", ".")))
    data = [float(x.strip().replace(",", ".")) for x in sample_distribution_file.read().splitlines()]

    phitter = PHITTER(data, fit_type="discrete")
    phitter.fit()
    print(phitter.not_rejected_results)

    for distribution, results in phitter.not_rejected_results.items():
        print(f"Distribution: {distribution}, SSE: {results['sse']}, Aprobados: {results['n_test_passed']}")
