import scipy.stats
import numpy

class TRIANGULAR:
    """
    Triangular distribution
    https://en.wikipedia.org/wiki/Triangular_distribution
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]
        self.c = self.parameters["c"]
        
    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        if x <= self.a:
            return 0
        if self.a < x and x <= self.c:
            return (x - self.a) ** 2 / ((self.b - self.a) * (self.c - self.a))
        if self.c < x and x < self.b:
            return 1 - ((self.b - x) ** 2 / ((self.b - self.a) * (self.b - self.c)))
        if x > self.b:
            return 1
    
    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        if x <= self.a:
            return 0
        if self.a <= x and x < self.c:
            return 2 * (x - self.a) / ((self.b - self.a) * (self.c - self.a))
        if x == self.c:
            return 2 / (self.b - self.a)
        if x > self.c and x <= self.b:
            return 2 * (self.b - x) / ((self.b - self.a) * (self.b - self.c))
        if x > self.b:
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
        v1 = self.a < self.c
        v2 = self.c < self.b
        return v1 and v2
        
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
            {"a": * , "b":  * , "c":  * }
        """
        ## Solve equations for estimation parameters
        # def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
        #     ## Variables declaration
        #     a, b, c = initial_solution
        #     print(measurements)
        #     ## Parametric expected expressions
        #     parametric_mean = (a + b + c) / 3
        #     parametric_variance = (a ** 2 + b ** 2 + c ** 2 - a * b - a * c - b * c) / 18
        #     parametric_skewness = math.sqrt(2) * (a + b - 2 * c) * (2 * a - b - c) * (a - 2 * b  + c) / (5 * (a ** 2 + b ** 2 + c ** 2 - a * b - a * c - b * c) ** (3 / 2))
            
        #     ## System Equations
        #     eq1 = parametric_mean - measurements.mean
        #     eq2 = parametric_variance - measurements.variance
        #     eq3 = parametric_skewness - measurements.skewness
        #     return (eq1, eq2, eq3)
        
        # solution = scipy.optimize.fsolve(equations, (1, 1, 1), measurements)
        
        ## Second method estimation
        a = measurements.min - 1e-3
        b = measurements.max + 1e-3
        c = 3 * measurements.mean - a - b
        
        ## Third method
        ## https://wernerantweiler.ca / blog.php?item=2019-06-05
        # q_1_16 = numpy.quantile(measurements.data, 1 / 16)
        # q_1_4 = numpy.quantile(measurements.data, 1 / 4)
        # q_3_4 = numpy.quantile(measurements.data, 3 / 4)
        # q_15_16 = numpy.quantile(measurements.data, 15 / 16)
        # u = (q_1_4 - q_1_16) ** 2
        # v = (q_15_16 - q_3_4) ** 2

        # a = 2 * q_1_16 - q_1_4
        # b = 2 * q_15_16 - q_3_4
        # c = (u * b + v * a) / (u + v)
        
        # Scipy parameters of distribution
        # scipy_params = scipy.stats.triang.fit(measurements.data)
        # a = scipy_params[1]
        # b = scipy_params[1] + scipy_params[2]
        # c = scipy_params[1] + scipy_params[2] * scipy_params[0]
        
        parameters = {"a": a, "b": b, "c": c}
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
    path = "../data/data_triangular.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS_CONTINUOUS(data)
    distribution = TRIANGULAR(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))