import numpy
import scipy.optimize
import scipy.stats


class DiscreteMeasures:
    def __init__(
        self,
        data: list[int],
        confidence_level: float = 0.95,
        subsample_size: int | None = None,
        subsample_estimation_size: int | None = None,
    ):
        self.data = numpy.sort(data) if subsample_size == None else numpy.sort(numpy.random.choice(data, size=subsample_size, replace=False))
        self.size = self.data.size
        self.data_to_fit = self.data if subsample_estimation_size == None else numpy.random.choice(self.data, size=min(self.size, subsample_estimation_size), replace=False)
        self.min = self.data[0]
        self.max = self.data[-1]
        self.mean = numpy.mean(data)
        self.variance = numpy.var(data, ddof=1)
        self.std = numpy.std(data, ddof=1)
        self.skewness = scipy.stats.moment(data, 3) / pow(self.std, 3)
        self.kurtosis = scipy.stats.moment(data, 4) / pow(self.std, 4)
        self.median = int(numpy.median(self.data))
        self.mode = int(scipy.stats.mode(data, keepdims=True)[0][0])
        self.domain = numpy.arange(self.min, self.max + 1)
        self.absolutes_frequencies, _ = numpy.histogram(self.data, bins=numpy.arange(self.min, self.max + 2) - 0.5, density=False)
        self.densities_frequencies, _ = numpy.histogram(self.data, bins=numpy.arange(self.min, self.max + 2) - 0.5, density=True)
        self.idx_ks = numpy.concatenate([numpy.where(self.data[:-1] != self.data[1:])[0], [self.size - 1]])
        self.Sn_ks = (numpy.arange(self.size) + 1) / self.size
        self.confidence_level = confidence_level
        self.critical_value_ks = scipy.stats.kstwo.ppf(self.confidence_level, self.size)
        self.ecdf_frequencies = numpy.cumsum(self.densities_frequencies)
        self.qq_arr = (numpy.arange(1, self.size + 1) - 0.5) / self.size

    def __str__(self) -> str:
        return str({"size": self.size, "mean": self.mean, "variance": self.variance, "skewness": self.skewness, "kurtosis": self.kurtosis, "median": self.median, "mode": self.mode})

    def __repr__(self) -> str:
        return str({"size": self.size, "mean": self.mean, "variance": self.variance, "skewness": self.skewness, "kurtosis": self.kurtosis, "median": self.median, "mode": self.mode})

    def get_dict(self) -> str:
        return {"size": self.size, "mean": self.mean, "variance": self.variance, "skewness": self.skewness, "kurtosis": self.kurtosis, "median": self.median, "mode": self.mode}

    def critical_value_chi2(self, freedom_degrees):
        return scipy.stats.chi2.ppf(self.confidence_level, freedom_degrees)


if __name__ == "__main__":
    ## Import function to get discrete_measures
    def get_data(path: str) -> list[int]:
        sample_distribution_file = open(path, "r")
        data = [int(x) for x in sample_distribution_file.read().splitlines()]
        sample_distribution_file.close()
        return data

    ## Distribution class
    path = "../discrete_distributions_sample/sample_geometric.txt"
    data = get_data(path)

    discrete_measures = DiscreteMeasures(data)

    print("size: ", discrete_measures.size)
    print("Min: ", discrete_measures.min)
    print("Max: ", discrete_measures.max)
    print("Mean: ", discrete_measures.mean)
    print("Variance: ", discrete_measures.variance)
    print("Skewness: ", discrete_measures.skewness)
    print("Kurtosis: ", discrete_measures.kurtosis)
    print("Median: ", discrete_measures.median)
    print("Mode: ", discrete_measures.mode)
    print("Domain: ", discrete_measures.domain)
    print("Absolutes Frequencies: ", discrete_measures.absolutes_frequencies)
    print("Densities Frequencies: ", discrete_measures.densities_frequencies)
