import math
import scipy.stats
import scipy.optimize
import numpy

class CAUCHY:
    """
    Cauchy distribution
    https://en.wikipedia.org/wiki/Cauchy_distribution     
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.x0 = self.parameters["x0"]
        self.gamma = self.parameters["gamma"]
        
    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        return (1 / math.pi) * math.atan(((x - self.x0) / self.gamma)) + (1 / 2)
    
    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        return 1 / (math.pi * self.gamma * (1 + ((x - self.x0) / self.gamma) ** 2))
    
    def get_num_parameters(self) -> int:
        """
        Number of parameters of the distribution
        """
        return len(self.parameters)
    
    def parameter_restrictions(self) -> bool:
        """
        Check parameters restrictions
        """
        v1 = self.gamma > 0
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
            {"x0":  * , "gamma":  * }
        """
        # ## First estimation
        # x0_ini = measurements.median
        # q1 = scipy.stats.scoreatpercentile(measurements.data, 25)
        # q3 = scipy.stats.scoreatpercentile(measurements.data, 75)
        # gamma_ini = (q3 - q1) / 2
        
        
        # ## Maximum Likelihood Estimation Cauchy distribution        
        # def objective(x):
        #     x0, gamma = x
        #     return - sum([math.log(1 / (math.pi * gamma * (1 + ((d - x0) / gamma) ** 2))) for d in measurements.data])
        # solution = scipy.optimize.minimize(objective, [x0_ini, gamma_ini], method="SLSQP", bounds = [(-numpy.inf, numpy.inf),(0,numpy.inf)])
        # print(solution)
        
        scipy_params = scipy.stats.cauchy.fit(measurements.data)
    
        ## Results
        parameters = {"x0": scipy_params[0], "gamma": scipy_params[1]}

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
    path = "../data/data_cauchy.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS_CONTINUOUS(data)
    distribution = CAUCHY(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    
    # import time
    # ti = time.time()
    # print(distribution.get_parameters(measurements))
    # print("Equations: ", time.time()  - ti)
    
    # ti = time.time()
    # print(scipy.stats.cauchy.fit(data))
    # print("Scipy: ",time.time()  - ti)
    