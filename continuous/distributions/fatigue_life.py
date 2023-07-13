import scipy.stats
import math
import scipy.optimize

class FATIGUE_LIFE:
    """
    Fatigue life distribution
    Also known as Birnbaum–Saunders distribution
    https://en.wikipedia.org/wiki/Birnbaum%E2%80%93Saunders_distribution
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        
        self.gamma = self.parameters["gamma"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]
        
    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        #result = scipy.stats.fatiguelife.cdf(x, self.gamma, loc=self.loc, scale=self.scale)
        z = lambda t: math.sqrt((t - self.loc) / self.scale)
        result = scipy.stats.norm.cdf((z(x) - 1 / z(x)) / (self.gamma))
        return result
    
    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        #result = scipy.stats.fatiguelife.pdf(x, self.gamma, loc=self.loc, scale=self.scale)
        z = lambda t: math.sqrt((t - self.loc) / self.scale)
        result = (z(x) + 1 / z(x)) / (2 * self.gamma * (x - self.loc)) * scipy.stats.norm.pdf((z(x) - 1 / z(x)) / (self.gamma))
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
        v1 = self.scale > 0
        v2 = self.gamma > 0
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
            {"gamma":  * , "loc":  * , "scale":  * }
        """
        ## NO SE ESTÁN RESOLVIENDO LAS ECUACIONES PARA GAMMA = 5, scale = 10, loc = 5
        # def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
        #     ## Variables declaration
        #     loc, scale, gamma = initial_solution
            
        #     ## Parametric expected expressions
        #     parametric_mean = loc + scale * (1 + gamma ** 2 / 2)
        #     parametric_variance = scale ** 2 * gamma ** 2 * (1 + 5 * gamma ** 2 / 4)
        #     parametric_skewness = 4 * gamma ** 2 * (11 * gamma ** 2 + 6) / ((4 + 5 * gamma ** 2) * math.sqrt(gamma ** 2 * (4 + 5 * gamma ** 2)))
        
        #     ## System Equations
        #     eq1 = parametric_mean - measurements.mean
        #     eq2 = parametric_variance - measurements.variance
        #     eq3 = parametric_skewness - measurements.skewness
            
        #     return (eq1, eq2, eq3)
        
        # solution = scipy.optimize.fsolve(equations, (1, 1, 1), measurements)
        # print(solution)
        scipy_params = scipy.stats.fatiguelife.fit(measurements.data)
        parameters = {"gamma": scipy_params[0], "loc": scipy_params[1], "scale": scipy_params[2]}
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
    path = "../data/data_fatigue_life.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = FATIGUE_LIFE(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))