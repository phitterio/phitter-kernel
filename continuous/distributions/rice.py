import scipy.special as sc
import math
import scipy.stats
import numpy
import scipy.optimize

class RICE:
    """
    Rice distribution
    https://en.wikipedia.org/wiki/Rice_distribution     
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.v = self.parameters["v"]
        self.sigma = self.parameters["sigma"]

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        def Q(M: float, a: float, b: float) -> float:
            """
            Marcum Q - function
            https://en.wikipedia.org/wiki/Marcum_Q - function
            """
            k = 1 - M
            x = (a / b) ** k * sc.iv(k, a * b)
            acum = 0
            while(x > 1e-20):
                acum += x
                k += 1
                x = (a / b) ** k * sc.iv(k, a * b)
            res= acum * math.exp(-(a ** 2 + b ** 2) / 2)
            return res

        # result = scipy.stats.rice.cdf(x, self.v / self.sigma, scale = self.sigma)
        # result, error = scipy.integrate.quad(self.pdf, 0, x)
        result = 1 - Q(1, self.v / self.sigma, x / self.sigma)
        return result
    
    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        # result = scipy.stats.rice.pdf(x, self.v / self.sigma, scale = self.sigma)
        result = (x / (self.sigma ** 2)) * math.exp(-(x ** 2 + self.v ** 2) / (2 * self.sigma ** 2)) * sc.i0(x * self.v / (self.sigma ** 2))
        return result

    def get_num_parameters(self) -> int:
        """
        Number of parameters of the distribution
        """
        return len(self.parameters)
    
    def parameter_restrictions(self) -> bool:
        """
        Check parameters restrictions
        """
        v1 = self.v > 0
        v2 = self.sigma > 0
        return v1 and v2
    
    def get_parameters(self, measurements) -> dict[str, float | int]:
        """
        Calculate proper parameters of the distribution from sample measurements.
        The parameters are calculated by solving the equations of the measures expected 
        for this distribution.The number of equations to consider is equal to the number 
        of parameters.
        
        Parameters
        ==========
        measurements: MEASUREMESTS
            attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, length, num_bins, data

        Returns
        =======
        parameters : dict
            {"alpha":  * , "beta":  * , "min":  * , "max":  * }
        """
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            v, sigma = initial_solution
            
            E = lambda k: sigma ** k * 2 ** (k / 2) * math.gamma(1 + k / 2) * sc.eval_laguerre(k / 2, - v * v / (2 * sigma * sigma))

            ## Parametric expected expressions
            parametric_mean = E(1)
            parametric_variance = (E(2) - E(1) ** 2)
            # parametric_skewness = (E(3) - 3 * E(2) * E(1) + 2 * E(1) ** 3) / ((E(2) - E(1) ** 2)) ** 1.5
            # parametric_kurtosis = (E(4)-4 * E(1) * E(3) + 6 * E(1) ** 2 * E(2) - 3 * E(1) ** 4) /  ((E(2) - E(1) ** 2)) ** 2
            
            ## System Equations
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            # eq3 = parametric_skewness - measurements.skewness
            # eq4 = parametric_kurtosis  - measurements.kurtosis
            
            
            return (eq1, eq2)
        
        bnds = ((0, 0), (numpy.inf, numpy.inf))
        x0 = (measurements.mean, math.sqrt(measurements.variance))
        args = ([measurements])
        solution = scipy.optimize.least_squares(equations, x0, bounds = bnds, args=args)
        parameters = {"v": solution.x[0], "sigma": solution.x[1]}
        
        
        return parameters


if __name__ == "__main__":   
    ## Import function to get measurements
    import sys
    sys.path.append("../measurements")
    from measurements import MEASUREMENTS_CONTINUOUS
    
    ## Import function to get measurements
    def get_data(direction):
        sample_distribution_file = open(direction, "r")
        data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
        return data
    
    ## Distribution class
    path = "../data/data_rice.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS_CONTINUOUS(data)
    distribution = RICE(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))