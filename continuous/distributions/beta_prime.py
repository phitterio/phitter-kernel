import scipy.optimize
import scipy.stats
import numpy
import scipy.special as sc

class BETA_PRIME:
    """
    Beta Prime Distribution
    https://en.wikipedia.org/wiki/Beta_prime_distribution
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]
        
    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        # result = scipy.stats.betaprime.cdf(x, self.alpha, self.beta)
        # print(result)
        result = sc.betainc(self.alpha, self.beta, x / (1 + x))
        return result
    
    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        # result = scipy.stats.betaprime.pdf(x, self.alpha, self.beta)
        # print(result)
        result = (x ** (self.alpha - 1) * (1 + x) ** (-self.alpha - self.beta)) / (sc.beta(self.alpha,self.beta))
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
            
            ## Variables declaration
            α, β = initial_solution
            
            ## Generatred moments function (not - centered)
            # E = lambda k: math.gamma(k - α) * math.gamma(β - k) / (math.gamma(α) * math.gamma(β))
            
        
            ## Parametric expected expressions
            parametric_mean = α / (β - 1)
            parametric_variance = α * (α + β - 1) / ((β - 1) ** 2 * (β - 2))
            # parametric_skewness = 2 * math.sqrt(((β - 2)) / (α * (α + β - 1))) * (((2 * α + β - 1)) / (β - 3))
            # parametric_kurtosis = (E(4)-4 * E(1) * E(3) + 6 * E(1) ** 2 * E(2) - 3 * E(1) ** 4) /  ((E(2) - E(1) ** 2)) ** 2
            # parametric_median = sc.betaincinv(0.5, α, β)
            # parametric_mode = (α - 1) / (β + 1)
        
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            # eq3 = parametric_skewness - measurements.skewness
            # eq2 = parametric_mode - measurements.mode
            
            return (eq1, eq2)

        
        scipy_params = scipy.stats.betaprime.fit(measurements.data)
        
        try:
            bnds = ((0, 0), (numpy.inf, numpy.inf))
            x0 = (scipy_params[0], scipy_params[1])
            args = ([measurements])
            solution = scipy.optimize.least_squares(equations, x0, bounds = bnds, args=args)
            parameters = {"alpha": solution.x[0], "beta": solution.x[1]}
        except:
            parameters = {"alpha": scipy_params[0], "beta": scipy_params[1]}
      
        return parameters

if __name__ == "__main__":
    ## Import function to get measurements
    import sys
    sys.path.append("../measurements")
    from measurements import MEASUREMENTS

    ## Import function to get measurements
    def get_data(direction):
        sample_distribution_file = open(direction, "r")
        data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
        return data
    
    ## Distribution class
    path = "../data/data_beta_prime.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = BETA_PRIME(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))