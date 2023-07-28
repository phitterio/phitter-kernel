import scipy.stats
import math
import scipy.special as sc
import numpy
import scipy.optimize

class INVERSE_GAMMA_3P:
    """
    Inverse Gamma distribution
    Also known Pearson Type 5 distribution
    https://en.wikipedia.org/wiki/Inverse - gamma_distribution
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]
        self.loc = self.parameters["loc"]
        
    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        ## Method 1: Integrate PDF function
        # result, error = scipy.integrate.quad(self.pdf, 0, x)
        # print(result)
        
        ## Method 2: Scipy Gamma Distribution class
        # result = scipy.stats.invgamma.cdf(x, a=self.alpha, scale=self.beta)
        # print(result)
        
        upper_inc_gamma = lambda a, x: sc.gammaincc(a, x) * math.gamma(a)
        result = upper_inc_gamma(self.alpha, self.beta / (x - self.loc)) / math.gamma(self.alpha)
        return result
    
    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        return ((self.beta ** self.alpha) * ((x - self.loc) ** (-self.alpha - 1)) * math.exp(-(self.beta / (x - self.loc)))) / math.gamma(self.alpha)
    
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
        The parameters are calculated by formula.
        
        Parameters
        ==========
        measurements: MEASUREMESTS
            attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, length, num_bins, data

        Returns
        =======
        parameters : dict
            {"alpha":  * , "beta":  * }
        """
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            ## Variables declaration
            α, β, loc = initial_solution
            
            ## Generatred moments function (not - centered)
            E = lambda k: (β ** k) / numpy.prod(numpy.array([(α - i) for i in range(1,k + 1)]))
            
            ## Parametric expected expressions
            parametric_mean = E(1) + loc
            parametric_variance = (E(2) - E(1) ** 2)
            parametric_skewness = (E(3) - 3 * E(2) * E(1) + 2 * E(1) ** 3) / ((E(2) - E(1) ** 2)) ** 1.5
            # parametric_kurtosis = (E(4)-4 * E(1) * E(3) + 6 * E(1) ** 2 * E(2) - 3 * E(1) ** 4) /  ((E(2) - E(1) ** 2)) ** 2
            
            ## System Equations
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            eq3 = parametric_skewness - measurements.skewness
            # eq4 = parametric_kurtosis  - measurements.kurtosis
            
            return (eq1, eq2, eq3)
        
        try:
            bnds = ((0, 0,  - numpy.inf), (numpy.inf, numpy.inf, numpy.inf))
            x0 = (2, 1, measurements.mean)
            args = ([measurements])
            solution = scipy.optimize.least_squares(equations, x0, bounds = bnds, args=args)
            parameters = {"alpha": solution.x[0], "beta": solution.x[1], "loc": solution.x[2]}
        except:
            scipy_params = scipy.stats.invgamma.fit(measurements.data)
            parameters = {"alpha": scipy_params[0], "beta": scipy_params[2], "loc": scipy_params[1]}
        
        return parameters
    



if __name__ == "__main__":  
    ## Import function to get measurements
    import sys
    sys.path.append("../measurements")
    from measurements import MEASUREMENTS
    
    ## Import function to get measurements
    def getData(direction):
        sample_distribution_file = open(direction, "r")
        data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
        return data
    
    ## Distribution class
    path = "../data/data_inverse_gamma_3p.txt"
    data = getData(path)
    measurements = MEASUREMENTS(data)
    distribution = INVERSE_GAMMA_3P(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))
    
