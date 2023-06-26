import math
import scipy.optimize
import numpy
import scipy.stats
import scipy.special as sc

class GENERALIZED_GAMMA:
    """
    Generalized Gamma Distribution
    https://en.wikipedia.org/wiki/Generalized_gamma_distribution
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        
        self.a = self.parameters["a"]
        self.d = self.parameters["d"]
        self.p = self.parameters["p"]

        
    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        # result = scipy.stats.gamma.cdf((x / self.a) ** self.p, a=self.d / self.p, scale=1)
        result = sc.gammainc(self.d / self.p, (x / self.a) ** self.p)
        
        return result
    
    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        return (self.p / (self.a ** self.d)) * (x ** (self.d - 1)) * math.exp(-(x / self.a) ** self.p) / math.gamma(self.d / self.p)
    
    def get_num_parameters(self) -> int:
        """
        Number of parameters of the distribution
        """
        return len(self.parameters)
    
    def parameter_restrictions(self) -> bool:
        """
        Check parameters restrictions
        """
        v1 = self.a > 0
        v2 = self.d > 0
        v3 = self.p > 0
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
            {"a":  * , "c":  * , "miu":  * , "sigma":  * }
        """
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            a, d, p = initial_solution
            
            E = lambda r: a ** r * (math.gamma((d + r) / p) / math.gamma(d / p))
            
            parametric_mean = E(1)
            parametric_variance = E(2) - E(1) ** 2
            parametric_skewness = (E(3) - 3 * E(2) * E(1) + 2 * E(1) ** 3) / ((E(2) - E(1) ** 2)) ** 1.5
            # parametric_kurtosis = (E(4)-4 * E(1) * E(3) + 6 * E(1) ** 2 * E(2) - 3 * E(1) ** 4) /  ((E(2) - E(1) ** 2)) ** 2
        
            ## System Equations
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            eq3 = parametric_skewness - measurements.skewness
            # eq4 = parametric_kurtosis  - measurements.kurtosis
        
            return (eq1, eq2, eq3)

        try:
            ## scipy.optimize.fsolve is 100x faster than least square but sometimes return solutions < 0
            solution = scipy.optimize.fsolve(equations, (1, 1, 1), measurements)
            
            ## If return a perameter < 0 then use least_square with restriction
            if all(x > 0 for x in solution) is False or all(x == 1 for x in solution) is True:
                bnds = ((0, 0, 0), (numpy.inf, numpy.inf, numpy.inf))
                x0 = (1, 1, 1)
                args = ([measurements])
                response = scipy.optimize.least_squares(equations, x0, bounds = bnds, args=args)
                solution = response.x
            parameters = {"a": solution[0], "d": solution[1], "p": solution[2]}
        except:
            scipy_params = scipy.stats.gengamma.fit(measurements.data)
            parameters = {"a": scipy_params[0], "c": scipy_params[1], "miu": scipy_params[2], "sigma": scipy_params[3]}

        return parameters
    
if __name__ == "__main__":
    ## Import function to get measurements
    from measurements_cont.measurements import MEASUREMENTS

    ## Import function to get measurements
    def get_data(direction):
        sample_distribution_file = open(direction, "r")
        data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
        return data
    
    ## Distribution class
    path = "../data/data_generalized_gamma.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = GENERALIZED_GAMMA(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))
    
    print(scipy.stats.gengamma.fit(measurements.data))
    
    
    