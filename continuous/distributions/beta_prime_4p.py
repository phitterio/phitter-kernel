import scipy.optimize
import scipy.stats
import numpy
import scipy.special as sc

class BETA_PRIME_4P:
    """
    scale Prime 4p Distribution
    https://en.wikipedia.org/wiki/Beta_prime_distribution
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]
        self.scale = self.parameters["scale"]
        self.loc = self.parameters["loc"]
        
    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        z = lambda t: (t - self.loc) / self.scale
        # result = scipy.stats.betaprime.cdf(x, self.alpha, self.beta, loc=self.loc, scale=self.scale)
        result = sc.betainc(self.alpha, self.beta, z(x) / (1 + z(x)))
        return result
    
    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        z = lambda t: (t - self.loc) / self.scale
        # result = scipy.stats.betaprime.pdf(x, self.alpha, self.beta, loc=self.loc, scale=self.scale)
        result = (1 / self.scale) * (z(x) ** (self.alpha - 1) * (1 + z(x)) ** (-self.alpha - self.beta)) / (sc.beta(self.alpha,self.beta))
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
        v1 = self.alpha > 0
        v2 = self.beta > 0
        v3 = self.scale > 0
        return v1 and v2 and v3
    
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
            {"alpha":  * , "scale":  * , "min":  * , "max":  * }
        """
        ## In this distribution solve the system is best than scipy estimation.
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            α, β, scale, loc = initial_solution
        
            parametric_mean = scale * α / (β - 1) + loc
            parametric_variance = (scale ** 2) * α * (α + β - 1) / ((β - 1) ** 2 * (β - 2))
            # parametric_skewness = 2 * math.sqrt(((β - 2)) / (α * (α + β - 1))) * (((2 * α + β - 1)) / (β - 3))
            parametric_median = loc + scale * scipy.stats.beta.ppf(0.5, α, β) / (1 - scipy.stats.beta.ppf(0.5, α, β))
            parametric_mode = scale * (α - 1) / (β + 1) + loc
            
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            # eq3 = parametric_skewness - measurements.skewness
            eq3 = parametric_median - measurements.median
            eq4 = parametric_mode - measurements.mode
            
            return (eq1, eq2, eq3, eq4)
        
        scipy_params = scipy.stats.betaprime.fit(measurements.data)
        
        try:
            bnds = ((0, 0, 0,  - numpy.inf), (numpy.inf, numpy.inf, numpy.inf, numpy.inf))
            x0 = (measurements.mean, measurements.mean, scipy_params[3], measurements.mean)
            args = ([measurements])
            solution = scipy.optimize.least_squares(equations, x0, bounds = bnds, args=args)
            parameters = {"alpha": solution.x[0], "beta": solution.x[1], "scale": solution.x[2], "loc": solution.x[3]}
        except:
            parameters = {"alpha": scipy_params[0], "beta": scipy_params[1], "scale": scipy_params[3], "loc": scipy_params[2]}
      
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
    path = "../data/data_beta_prime_4p.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS_CONTINUOUS(data)
    distribution = BETA_PRIME_4P(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))