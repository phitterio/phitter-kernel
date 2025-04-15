import numpy
import scipy.optimize
import scipy.stats


class ContinuousMeasures:
    def __init__(
        self,
        data: list[int | float] | numpy.ndarray,
        num_bins: int | None = None,
        confidence_level: float = 0.95,
        subsample_size: int | None = None,
        subsample_estimation_size: int | None = None,
    ):
        self.data = numpy.sort(data) if subsample_size == None else numpy.sort(numpy.random.choice(data, size=subsample_size, replace=False))
        self.data_unique = numpy.unique(self.data)
        self.size = self.data.size
        self.data_to_fit = self.data if subsample_estimation_size == None else numpy.random.choice(self.data, size=min(self.size, subsample_estimation_size), replace=False)
        self.min = self.data[0]
        self.max = self.data[-1]
        self.mean = numpy.mean(self.data)
        self.variance = numpy.var(self.data, ddof=1)
        self.standard_deviation = numpy.sqrt(self.variance)
        self.skewness = scipy.stats.skew(self.data, bias=False)
        self.kurtosis = scipy.stats.kurtosis(self.data, fisher=False, bias=False)
        self.median = numpy.median(self.data)
        self.mode = self.calculate_mode()
        self.num_bins = num_bins if num_bins != None else len(numpy.histogram_bin_edges(self.data, bins="doane"))
        self.absolutes_frequencies, self.bin_edges = numpy.histogram(self.data, self.num_bins)
        self.densities_frequencies, _ = numpy.histogram(self.data, self.num_bins, density=True)
        self.central_values = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        self.idx_ks = numpy.concatenate([numpy.where(self.data[:-1] != self.data[1:])[0], [self.size - 1]])
        self.Sn_ks_before = (numpy.arange(self.size)) / self.size
        self.Sn_ks_after = (numpy.arange(self.size) + 1) / self.size
        self.confidence_level = confidence_level
        self.critical_value_ks = scipy.stats.kstwo.ppf(self.confidence_level, self.size)
        self.critical_value_ad = self.ad_critical_value(self.confidence_level, self.size)
        self.ecdf_frequencies = numpy.searchsorted(self.data, self.data_unique, side="right") / self.data.size
        self.qq_arr = (numpy.arange(1, self.size + 1) - 0.5) / self.size

    def __str__(self) -> str:
        return str({"size": self.size, "mean": self.mean, "variance": self.variance, "skewness": self.skewness, "kurtosis": self.kurtosis, "median": self.median, "mode": self.mode})

    def __repr__(self) -> str:
        return str({"size": self.size, "mean": self.mean, "variance": self.variance, "skewness": self.skewness, "kurtosis": self.kurtosis, "median": self.median, "mode": self.mode})

    def get_dict(self) -> str:
        return {"size": self.size, "mean": self.mean, "variance": self.variance, "skewness": self.skewness, "kurtosis": self.kurtosis, "median": self.median, "mode": self.mode}

    def calculate_mode(self) -> float:
        distribution = scipy.stats.gaussian_kde(self.data)
        x = numpy.linspace(self.min, self.max, 10000)
        y = distribution.pdf(x)
        return x[numpy.argmax(y)]
        # solution = scipy.optimize.shgo(lambda x: -distribution.pdf(x)[0], bounds=[(self.min, self.max)], n=100 * self.size)
        # solution = scipy.optimize.minimize(lambda x: -distribution.pdf(x)[0], x0=[self.mean], bounds=[(self.min, self.max)])
        # return solution.x[0]

    def critical_value_chi2(self, freedom_degrees: int):
        return scipy.stats.chi2.ppf(self.confidence_level, freedom_degrees)

    def adinf(self, z: float):
        if z < 2:
            return (z**-0.5) * numpy.exp(-1.2337141 / z) * (2.00012 + (0.247105 - (0.0649821 - (0.0347962 - (0.011672 - 0.00168691 * z) * z) * z) * z) * z)
        return numpy.exp(-numpy.exp(1.0776 - (2.30695 - (0.43424 - (0.082433 - (0.008056 - 0.0003146 * z) * z) * z) * z) * z))

    def errfix(self, n: float, x: float) -> float:
        def g1(t: float) -> float:
            return numpy.sqrt(t) * (1 - t) * (49 * t - 102)

        def g2(t: float) -> float:
            return -0.00022633 + (6.54034 - (14.6538 - (14.458 - (8.259 - 1.91864 * t) * t) * t) * t) * t

        def g3(t: float) -> float:
            return -130.2137 + (745.2337 - (1705.091 - (1950.646 - (1116.360 - 255.7844 * t) * t) * t) * t) * t

        c = 0.01265 + 0.1757 / n
        if x < c:
            return (0.0037 / (n**3) + 0.00078 / (n**2) + 0.00006 / n) * g1(x / c)
        elif x > c and x < 0.8:
            return (0.04213 / n + 0.01365 / (n**2)) * g2((x - c) / (0.8 - c))
        else:
            return (g3(x)) / n

    def AD(self, n: float, z: float) -> float:
        return self.adinf(z) + self.errfix(n, self.adinf(z))

    def ad_critical_value(self, q: float, n: float) -> float:
        f = lambda x: self.AD(n, x) - q
        root = scipy.optimize.newton(f, 2)
        return root

    def ad_p_value(self, n: float, z: float) -> float:
        return 1 - self.AD(n, z)


if __name__ == "__main__":
    ## Import function to get continuous_measures
    def get_data(path: str) -> list[float]:
        sample_distribution_file = open(path, "r")
        data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
        sample_distribution_file.close()
        return data

    ## Distribution class
    path = "../continuous_distributions_sample/sample_exponential.txt"
    data = get_data(path)

    continuous_measures = ContinuousMeasures(data)

    print(f"Size: {continuous_measures.size}")
    print(f"Min: {continuous_measures.min}")
    print(f"Max: {continuous_measures.max}")
    print(f"Mean: {continuous_measures.mean}")
    print(f"Variance: {continuous_measures.variance}")
    print(f"Standard Deviation: {continuous_measures.standard_deviation}")
    print(f"Skewness: {continuous_measures.skewness}")
    print(f"Kurtosis: {continuous_measures.kurtosis}")
    print(f"Median: {continuous_measures.median}")
    print(f"Mode: {continuous_measures.mode}")
    print(f"Num bins: {continuous_measures.num_bins}")
    print(f"Central Values: {continuous_measures.central_values}")
    print(f"Critical value AD: {continuous_measures.critical_value_ad}")
