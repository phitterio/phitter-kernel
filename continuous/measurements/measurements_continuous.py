import scipy.stats
import numpy
import scipy.optimize
import math


class MEASUREMENTS_CONTINUOUS:
    def __init__(self, data: list[float | int], num_bins: int | None = None):
        self.data = data
        self.length = len(data)
        self.min = min(data)
        self.max = max(data)
        self.mean = numpy.mean(data)
        self.variance = numpy.var(data, ddof=1)
        self.standard_deviation = numpy.std(data, ddof=1)
        self.skewness = scipy.stats.moment(data, 3) / pow(self.standard_deviation, 3)
        self.kurtosis = scipy.stats.moment(data, 4) / pow(self.standard_deviation, 4)
        self.median = numpy.median(data)
        self.mode = self.calculate_mode()
        self.num_bins = num_bins if num_bins != None else self.num_bins_doane()

    def __str__(self) -> str:
        return str({"length": self.length, "mean": self.mean, "variance": self.variance, "skewness": self.skewness, "kurtosis": self.kurtosis, "median": self.median, "mode": self.mode})

    def __repr__(self) -> str:
        return str({"length": self.length, "mean": self.mean, "variance": self.variance, "skewness": self.skewness, "kurtosis": self.kurtosis, "median": self.median, "mode": self.mode})

    def get_dict(self) -> str:
        return {"length": self.length, "mean": self.mean, "variance": self.variance, "skewness": self.skewness, "kurtosis": self.kurtosis, "median": self.median, "mode": self.mode}

    def calculate_mode(self) -> float:
        def calc_shgo_mode(distribution):
            objective = lambda x: -distribution.pdf(x)[0]
            bnds = [[self.min, self.max]]
            solution = scipy.optimize.shgo(objective, bounds=bnds, n=100 * self.length)
            return solution.x[0]

        ## KDE
        distribution = scipy.stats.gaussian_kde(self.data)
        shgo_mode = calc_shgo_mode(distribution)
        return shgo_mode

    def num_bins_doane(self):
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
        skewness = scipy.stats.skew(self.data)
        sigma_g1 = math.sqrt((6 * (N - 2)) / ((N + 1) * (N + 3)))
        num_bins = 1 + math.log(N, 2) + math.log(1 + abs(skewness) / sigma_g1, 2)
        num_bins = round(num_bins)
        return num_bins


if __name__ == "__main__":
    ## Import function to get measurements
    def get_data(direction):
        sample_distribution_file = open(direction, "r")
        data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
        return data

    ## Distribution class
    path = "./data/data_generalized_pareto.txt"
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
