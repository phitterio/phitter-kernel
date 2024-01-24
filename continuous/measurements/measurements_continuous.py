import scipy.stats
import math
import scipy.optimize
import numpy


class MEASUREMENTS_CONTINUOUS:
    def __init__(
        self,
        data: list[float | int],
        num_bins: int | None = None,
        confidence_level: float = 0.95,
    ):
        self.data = numpy.sort(data)
        self.length = len(self.data)
        self.min = self.data[0]
        self.max = self.data[-1]
        self.mean = numpy.mean(self.data)
        self.variance = numpy.var(self.data, ddof=1)
        self.standard_deviation = numpy.std(self.data, ddof=1)
        self.skewness = scipy.stats.moment(self.data, 3) / pow(self.standard_deviation, 3)
        self.kurtosis = scipy.stats.moment(self.data, 4) / pow(self.standard_deviation, 4)
        self.median = numpy.median(self.data)
        self.mode = self.calculate_mode()
        self.num_bins = num_bins if num_bins != None else self.num_bins_doane(self.data)
        self.absolutes_frequencies, self.bin_edges = numpy.histogram(self.data, self.num_bins)
        self.densities_frequencies, _ = numpy.histogram(self.data, self.num_bins, density=True)
        self.central_values = [(self.bin_edges[i] + self.bin_edges[i + 1]) / 2 for i in range(len(self.bin_edges) - 1)]
        self.idx_ks = numpy.concatenate([numpy.where(self.data[:-1] != self.data[1:])[0], [self.length - 1]])
        self.Sn_ks = (numpy.arange(self.length) + 1) / self.length
        self.critical_value_ad = self.ad_critical_value(confidence_level, self.length)

    def __str__(self) -> str:
        return str({"length": self.length, "mean": self.mean, "variance": self.variance, "skewness": self.skewness, "kurtosis": self.kurtosis, "median": self.median, "mode": self.mode})

    def __repr__(self) -> str:
        return str({"length": self.length, "mean": self.mean, "variance": self.variance, "skewness": self.skewness, "kurtosis": self.kurtosis, "median": self.median, "mode": self.mode})

    def get_dict(self) -> str:
        return {"length": self.length, "mean": self.mean, "variance": self.variance, "skewness": self.skewness, "kurtosis": self.kurtosis, "median": self.median, "mode": self.mode}

    def calculate_mode(self) -> float:
        distribution = scipy.stats.gaussian_kde(self.data)
        solution = scipy.optimize.shgo(lambda x: -distribution.pdf(x)[0], bounds=[[self.min, self.max]], n=100 * self.length)
        return solution.x[0]

    def num_bins_doane(self, data):
        """
        DONAE'S FORMULA
        https://en.wikipedia.org/wiki/Histogram#Doane's_formula

        Parameters
        ==========
        data : iterable
            data set
        Returns
        =======
        num_bins : int
            Cumulative distribution function evaluated at x
        """
        N = self.length
        skewness = scipy.stats.skew(data)
        sigma_g1 = numpy.sqrt((6 * (N - 2)) / ((N + 1) * (N + 3)))
        num_bins = 1 + numpy.log2(N) + numpy.log2(1 + abs(skewness) / sigma_g1)
        return math.ceil(num_bins)

    def adinf(self, z):
        if z < 2:
            return (z**-0.5) * numpy.exp(-1.2337141 / z) * (2.00012 + (0.247105 - (0.0649821 - (0.0347962 - (0.011672 - 0.00168691 * z) * z) * z) * z) * z)
        return numpy.exp(-numpy.exp(1.0776 - (2.30695 - (0.43424 - (0.082433 - (0.008056 - 0.0003146 * z) * z) * z) * z) * z))

    def errfix(self, n, x):
        def g1(t):
            return numpy.sqrt(t) * (1 - t) * (49 * t - 102)

        def g2(t):
            return -0.00022633 + (6.54034 - (14.6538 - (14.458 - (8.259 - 1.91864 * t) * t) * t) * t) * t

        def g3(t):
            return -130.2137 + (745.2337 - (1705.091 - (1950.646 - (1116.360 - 255.7844 * t) * t) * t) * t) * t

        c = 0.01265 + 0.1757 / n
        if x < c:
            return (0.0037 / (n**3) + 0.00078 / (n**2) + 0.00006 / n) * g1(x / c)
        elif x > c and x < 0.8:
            return (0.04213 / n + 0.01365 / (n**2)) * g2((x - c) / (0.8 - c))
        else:
            return (g3(x)) / n

    def AD(self, n, z):
        return self.adinf(z) + self.errfix(n, self.adinf(z))

    def ad_critical_value(self, q, n):
        f = lambda x: self.AD(n, x) - q
        root = scipy.optimize.newton(f, 2)
        return root

    def ad_p_value(self, n, z):
        return 1 - self.AD(n, z)


if __name__ == "__main__":
    ## Import function to get measurements
    def get_data(direction):
        sample_distribution_file = open(direction, "r")
        data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
        return data

    ## Distribution class
    path = "../data/data_generalized_pareto.txt"
    data = get_data(path)

    measurements = MEASUREMENTS_CONTINUOUS(data)

    print("Length: ", measurements.length)
    print("Min: ", measurements.min)
    print("Max: ", measurements.max)
    print("Mean: ", measurements.mean)
    print("Variance: ", measurements.variance)
    print("Skewness: ", measurements.skewness)
    print("Kurtosis: ", measurements.kurtosis)
    print("Median: ", measurements.median)
    print("Mode: ", measurements.mode)
    print("Num bins", measurements.num_bins)
    print("Central Values", measurements.central_values)
    print("Critical value AD", measurements.critical_value_ad)
