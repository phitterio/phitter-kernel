import scipy.stats
import numpy
import scipy.optimize
import collections


class MEASUREMENTS_DISCRETE:
    def __init__(self, data):
        self.data = data
        self.length = len(data)
        self.min = min(data)
        self.max = max(data)
        self.mean = numpy.mean(data)
        self.variance = numpy.var(data, ddof=1)
        self.std = numpy.std(data, ddof=1)
        self.skewness = scipy.stats.moment(data, 3) / pow(self.std, 3)
        self.kurtosis = scipy.stats.moment(data, 4) / pow(self.std, 4)
        self.median = int(numpy.median(self.data))
        self.mode = int(scipy.stats.mode(data, keepdims=True)[0][0])
        self.frequencies = self.get_frequencies()

    def __str__(self) -> str:
        return str({"length": self.length, "mean": self.mean, "variance": self.variance, "skewness": self.skewness, "kurtosis": self.kurtosis, "median": self.median, "mode": self.mode})

    def __repr__(self) -> str:
        return str({"length": self.length, "mean": self.mean, "variance": self.variance, "skewness": self.skewness, "kurtosis": self.kurtosis, "median": self.median, "mode": self.mode})

    def get_dict(self) -> str:
        return {"length": self.length, "mean": self.mean, "variance": self.variance, "skewness": self.skewness, "kurtosis": self.kurtosis, "median": self.median, "mode": self.mode}

    def get_frequencies(self) -> dict[int, float | int]:
        frequencies = collections.Counter(self.data)
        return {k: v for k, v in sorted(frequencies.items(), key=lambda item: item[0])}


if __name__ == "__main__":
    ## Import function to get measurements
    def get_data(direction):
        sample_distribution_file = open(direction, "r")
        data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
        return data

    ## Distribution class
    path = "../data/data_binomial.txt"
    data = get_data(path)

    measurements = MEASUREMENTS_DISCRETE(data)

    print("Length: ", measurements.length)
    print("Min: ", measurements.min)
    print("Max: ", measurements.max)
    print("Mean: ", measurements.mean)
    print("Variance: ", measurements.variance)
    print("Skewness: ", measurements.skewness)
    print("Kurtosis: ", measurements.kurtosis)
    print("Median: ", measurements.median)
    print("Mode: ", measurements.mode)
    print(measurements.frequencies)
