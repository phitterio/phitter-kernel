import scipy.stats
import numpy
import scipy.optimize

class GENERALIZED_PARETO:
    """
    Generalized Pareto distribution
    https://en.wikipedia.org/wiki/Generalized_Pareto_distribution       
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.c = self.parameters["c"]
        self.miu = self.parameters["miu"]
        self.sigma = self.parameters["sigma"]
        
    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        
        # result = scipy.stats.genpareto.cdf(x, self.c, loc = self.miu, scale = self.sigma)
        z = lambda t: (t - self.miu) / self.sigma
        result = 1 - (1 + self.c * z(x)) ** (-1 / self.c)
        return result
    
    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        # result = scipy.stats.genpareto.pdf(x, self.c, loc = self.miu, scale = self.sigma)
        z = lambda t: (t - self.miu) / self.sigma
        result = (1 / self.sigma) * (1 + self.c * z(x)) ** (-1 / self.c - 1)
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
        v1 = self.sigma > 0
        return v1

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
            {"c":  * , "miu":  * , "sigma":  * }
        """        
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            
            ## Variables declaration
            c, miu, sigma = initial_solution
            
            ## Parametric expected expressions
            parametric_mean = miu + sigma / (1 - c)
            parametric_variance = sigma * sigma / ((1 - c) * (1 - c) * (1 - 2 * c))
            # parametric_skewness = 2 * (1 + c) * math.sqrt(1 - 2 * c) / (1 - 3 * c)
            # parametric_kurtosis = 3 * (1 - 2 * c) * (2 * c * c + c + 3) / ((1 -  3 * c) * (1 -  4 + c))
            parametric_median = miu + sigma * (2 ** c - 1) / c
            # parametric_mode = loc
            
            # System Equations
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            # eq1 = parametric_skewness - measurements.skewness
            # eq3 = parametric_kurtosis - measurements.kurtosis
            # eq1 = parametric_p25 - numpy.percentile(measurements.data, 25)
            eq3 = parametric_median - numpy.percentile(measurements.data, 50)
            
            return (eq1, eq2, eq3)
        
        ## The scipy genpareto.fit is good from samples whit c > 0
        ## but it's not so much when c < 0.
        ## The solution of system of equation is so good. The problem is that
        ## the measurements of genpareto distributions is not defined from c >  1 / 4 (kurtosis)
        scipy_params = scipy.stats.genpareto.fit(measurements.data)
        parameters = {"c": scipy_params[0], "miu": scipy_params[1], "sigma": scipy_params[2]}
        
        if parameters["c"] < 0:
            scipy_params = scipy.stats.genpareto.fit(measurements.data)
            c0 = scipy_params[0]
            x0 = [c0, measurements.min, 1]
            b = ((-numpy.inf,  - numpy.inf, 0), (numpy.inf, measurements.min, numpy.inf))
            solution = scipy.optimize.least_squares(equations, x0, bounds = b, args=([measurements]))
            parameters = {"c": solution.x[0], "miu": solution.x[1], "sigma": solution.x[2]}
            
            ## When c < 0 the domain of of x is [miu, miu - sigma / c]
            ## Forthis reason miu < measurements.min and miu - sigma / c < measurements.max
            parameters["miu"] = min(parameters["miu"], measurements.min - 1e-3)
            delta_sigma = parameters["c"] * (parameters["miu"] - measurements.max) - parameters["sigma"]
            parameters["sigma"] = parameters["sigma"] + delta_sigma + 1e-8
            # print(parameters["miu"], parameters["miu"] - parameters["sigma"] / parameters["c"])
            
      
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
    path = "../data/data_generalized_pareto.txt"
    data = get_data(path)
    measurements = MEASUREMENTS(data)
    distribution = GENERALIZED_PARETO(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))
