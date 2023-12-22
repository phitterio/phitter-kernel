import scipy.optimize


class TRAPEZOIDAL:
    """
    Trapezoidal distribution
    https://en.wikipedia.org/wiki/Trapezoidal_distribution
    """

    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]
        self.c = self.parameters["c"]
        self.d = self.parameters["d"]

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        if x <= self.a:
            return 0
        if self.a <= x and x < self.b:
            return (1 / (self.d + self.c - self.b - self.a)) * (1 / (self.b - self.a)) * (x - self.a) ** 2
        if self.b <= x and x < self.c:
            return (1 / (self.d + self.c - self.b - self.a)) * (2 * x - self.a - self.b)
        if self.c <= x and x <= self.d:
            return 1 - (1 / (self.d + self.c - self.b - self.a)) * (1 / (self.d - self.c)) * (self.d - x) ** 2
        if x >= self.d:
            return 1

    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        if x <= self.a:
            return 0
        if self.a <= x and x < self.b:
            return (2 / (self.d + self.c - self.b - self.a)) * ((x - self.a) / (self.b - self.a))
        if self.b <= x and x < self.c:
            return 2 / (self.d + self.c - self.b - self.a)
        if self.c <= x and x <= self.d:
            return (2 / (self.d + self.c - self.b - self.a)) * ((self.d - x) / (self.d - self.c))
        if x >= self.d:
            return 0

    def get_num_parameters(self) -> int:
        """
        Number of parameters of the distribution
        """
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        """
        Check parameters restrictions
        """
        v1 = self.a < self.b
        v2 = self.b < self.c
        v3 = self.c < self.d
        return v1 and v2 and v3

    def get_parameters(self, measurements) -> dict[str, float | int]:
        """
        Calculate proper parameters of the distribution from sample measurements.
        The parameters are calculated by formula.

        Parameters
        ==========
        measurements: MEASUREMESTS
            attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, length, num_bins, data

        Returns
        =======
        parameters : dict
            {"a": * , "b": * , "c": * }
        """

        a = measurements.min - 1e-3
        d = measurements.max + 1e-3

        def equations(initial_solution, measurements, a, d):
            ## Variables declaration
            b, c = initial_solution

            ## Parametric expected expressions
            parametric_mean = (1 / (3 * (d + c - a - b))) * ((d**3 - c**3) / (d - c) - (b**3 - a**3) / (b - a))
            parametric_variance = (1 / (6 * (d + c - a - b))) * ((d**4 - c**4) / (d - c) - (b**4 - a**4) / (b - a)) - (
                (1 / (3 * (d + c - a - b))) * ((d**3 - c**3) / (d - c) - (b**3 - a**3) / (b - a))
            ) ** 2

            ## System Equations
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance

            return (eq1, eq2)

        x0 = [(d + a) * 0.25, (d + a) * 0.75]
        bnds = ((a, a), (d, d))
        solution = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=([measurements, a, d]))

        parameters = {"a": a, "b": solution.x[0], "c": solution.x[1], "d": d}
        return parameters


if __name__ == "__main__":
    ## Import function to get measurements
    import sys

    sys.path.append("../measurements")
    from measurements_continuous import MEASUREMENTS_CONTINUOUS

    ## Import function to get measurements
    def get_data(direction):
        sample_distribution_file = open(direction, "r")
        data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
        return data

    ## Distribution class
    path = "../data/data_trapezoidal.txt"
    data = get_data(path)
    measurements = MEASUREMENTS_CONTINUOUS(data)
    distribution = TRAPEZOIDAL(measurements)

    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))
