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
    def __init__(self, data: list[int | float], fit_type="continuous"):
        self.data = data
        self.fit_type = fit_type
        self.sorted_results_sse = None
        self.not_rejected_results = None

    def fit(self):
        if self.fit_type == "continuous":
            phitter_continuous = PHITTER_CONTINUOUS(self.data)
            self.sorted_results_sse, self.not_rejected_results = phitter_continuous.fit()

        if self.fit_type == "discrete":
            phitter_discrete = PHITTER_DISCRETE(self.data)
            self.sorted_results_sse, self.not_rejected_results = phitter_discrete.fit()

if __name__ == "__main__":
    path = "../continuous/data/data_beta.txt"
    sample_distribution_file = open(path, "r")
    data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]

    phitter = PHITTER(data)
    phitter.fit()
    print(phitter.not_rejected_results)