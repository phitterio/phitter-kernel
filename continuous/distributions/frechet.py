import scipy.stats
import math
import scipy.optimize

class FRECHET:
    """
    Fréchet distribution
    Also known as inverse Weibull distribution (Scipy name)
    https://en.wikipedia.org/wiki/Fr%C3%A9chet_distribution
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        
        self.alpha = self.parameters["alpha"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]
        
    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        z = lambda t: (t - self.loc) / self.scale
        # result = scipy.stats.invweibull.pdf(x, self.c, loc = self.loc, scale = self.scale)
        result = (1 / self.scale) * self.alpha * z(x) ** (-self.alpha - 1) * math.exp(-z(x) **  - self.alpha)
        return result
    
    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        # print(scipy.stats.invweibull.pdf(40.89022608, self.alpha, loc = self.loc, scale = self.scale))
        return (self.alpha / self.scale) * (((x - self.loc) / self.scale) ** (-1 - self.alpha)) * math.exp(-((x - self.loc) / self.scale) ** (-self.alpha))
    
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
        v2 = self.scale > 0
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
            {"alpha":  * , "m":  * , "s":  * }
        """
        scipy_params = scipy.stats.invweibull.fit(measurements.data)
        parameters = {"alpha": scipy_params[0], "loc": scipy_params[1], "scale": scipy_params[2]}
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
    path = "../data/data_frechet.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = FRECHET(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))