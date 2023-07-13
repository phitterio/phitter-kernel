import scipy.stats
import scipy.special as sc
import scipy.optimize
import numpy
import math


class LOGGAMMA:
    """
    Loggamma distribution
    https://docs.scipy.org / doc / scipy / reference / generated / scipy.stats.loggamma.html     
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
        # result = scipy.stats.gamma.cdf(math.exp((x - self.miu) / self.sigma), a=self.c, scale=1)
        # print(scipy.stats.loggamma.cdf(x, self.c, loc=self.miu, scale=self.sigma))
        
        y = lambda x: (x - self.miu) / self.sigma
        result = sc.gammainc(self.c, math.exp(y(x)))
        return result
    
    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        # print(scipy.stats.loggamma.pdf(x, self.c, loc=self.miu, scale=self.sigma))
        y = lambda x: (x - self.miu) / self.sigma
        result = math.exp(self.c * y(x) - math.exp(y(x)) - sc.gammaln(self.c)) / self.sigma
        
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
        v1 = self.c > 0
        v2 = self.sigma > 0
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
            {"c":  * , "miu":  * , "sigma":  * }
        """
        def equations(initial_solution, data_mean, data_variance, data_skewness):
            c, miu, sigma = initial_solution
            
            # parametric_mean, parametric_variance, parametric_skewness, parametric_kurtosis = scipy.stats.loggamma.stats(c, loc=miu, scale=sigma, moments='mvsk')
            parametric_mean = sc.digamma(c) * sigma + miu
            parametric_variance = sc.polygamma(1, c) * (sigma ** 2)
            parametric_skewness = sc.polygamma(2, c) / (sc.polygamma(1, c) ** 1.5)
            # parametric_kurtosis = sc.polygamma(3, c) / (sc.polygamma(1, c) ** 2)
            
            ## System Equations
            eq1 = parametric_mean - data_mean
            eq2 = parametric_variance - data_variance
            eq3 = parametric_skewness - data_skewness
            # eq4 = parametric_kurtosis  - measurements.kurtosis
            
            return (eq1, eq2, eq3)
    
        bnds = ((0, 0, 0), (numpy.inf, numpy.inf, numpy.inf))
        x0 = (1,1,1)
        args = (measurements.mean, measurements.variance, measurements.skewness)
        solution = scipy.optimize.least_squares(equations, x0, bounds = bnds, args=args)
        parameters = {"c": solution.x[0], "miu": solution.x[1], "sigma": solution.x[2]}
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
    path = "../data/data_loggamma.txt"
    data = get_data(path)
    measurements = MEASUREMENTS(data)
    distribution = LOGGAMMA(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))
    
    print("========= Time parameter estimation analisys ========")
    
    import time
    
    def equations(initial_solution, data_mean, data_variance, data_skewness):
        c, miu, sigma = initial_solution
        parametric_mean, parametric_variance, parametric_skewness, parametric_kurtosis = scipy.stats.loggamma.stats(c, loc=miu, scale=sigma, moments='mvsk')
        parametric_mean = sc.digamma(c) * sigma + miu
        parametric_variance = sc.polygamma(1, c) * (sigma ** 2)
        parametric_skewness = sc.polygamma(2, c) / (sc.polygamma(1, c) ** 1.5)
        # parametric_kurtosis = sc.polygamma(3, c) / (sc.polygamma(1, c) ** 2)
        
        ## System Equations
        eq1 = parametric_mean - data_mean
        eq2 = parametric_variance - data_variance
        eq3 = parametric_skewness - data_skewness
        # eq4 = parametric_kurtosis  - measurements.kurtosis
        
        return (eq1, eq2, eq3)
    
    ti = time.time()
    bnds = ((0, 0, 0), (numpy.inf, numpy.inf, numpy.inf))
    x0 = (1,1,1)
    args = (measurements.mean, measurements.variance, measurements.skewness)
    solution = scipy.optimize.least_squares(equations, x0, bounds = bnds, args=args)
    parameters = {"c": solution.x[0], "miu": solution.x[1], "sigma": solution.x[2]}
    print(parameters)
    print("Solve equations time: ", time.time()  - ti)
    
    ti = time.time()
    scipy_params = scipy.stats.loggamma.fit(data)
    parameters = {"c": scipy_params[0], "miu": scipy_params[1], "sigma": scipy_params[2]}
    print(parameters)
    print("Scipy time get parameters: ",time.time()  - ti)

