import scipy.integrate
import math
import scipy.stats

class JOHNSON_SB:
    """
    Johnson SB distribution
    http://www.ntrand.com/johnson-sb-distribution           
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.xi_ = self.parameters["xi"]
        self.lambda_ = self.parameters["lambda"]
        self.gamma_ = self.parameters["gamma"]
        self.delta_ = self.parameters["delta"]
    
    def cdf(self, x: float) -> float:      
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        # result, error = scipy.integrate.quad(self.pdf, self.xi_, x)
        z = lambda t: (t - self.xi_) / self.lambda_
        result = scipy.stats.norm.cdf(self.gamma_ + self.delta_ * math.log(z(x) / (1 - z(x))))
        return result
    
    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        z = lambda t: (t - self.xi_) / self.lambda_
        return (self.delta_ / (self.lambda_ * math.sqrt(2 * math.pi) * z(x) * (1 - z(x)))) * math.e ** (-(1 / 2) * (self.gamma_ + self.delta_ * math.log(z(x) / (1 - z(x)))) ** 2)
    
    def get_num_parameters(self) -> int:
        """
        Number of parameters of the distribution
        """
        return len(self.parameters)
    
    def parameter_restrictions(self) -> bool:
        """
        Check parameters restrictions
        """
        v1 = self.delta_ > 0
        v2 = self.lambda_ > 0
        return v1 and v2

    def get_parameters(self, measurements) -> dict[str, float | int]:
        """
        Calculate proper parameters of the distribution from sample measurements.
        The parameters are calculated with the method proposed in [1].
        
        Parameters
        ==========
        measurements: MEASUREMESTS
            attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, length, num_bins, data

        Returns
        =======
        parameters : dict
            {"xi":  * , "lambda":  * , "gamma":  * , "delta":  * }
        
        References
        ==========
        .. [1] George, F., & Ramachandran, K. M. (2011). 
               Estimation of parameters of Johnson’s system of distributions. 
               Journal of Modern Applied Statistical Methods, 10(2), 9.
        .. [2] https://www.nature.com / articles / 7500437
        """
        # ## Percentiles
        # z = 0.5384
        # percentiles = [scipy.stats.norm.cdf(0.5384 * i) for i in range(-3, 4, 2)]
        # x1, x2, x3, x4 = [scipy.stats.scoreatpercentile(measurements.data, 100 * x) for x in percentiles]
        
        # ## Calculation m,n,p
        # m = x4 - x3
        # n = x2 - x1
        # p = x3 - x2
        
        # ## Calculation distribution parameters
        # lambda_ = (p * math.sqrt((((1 + p / m) * (1 + p / n) - 2) ** 2-4))) / (p ** 2 / (m * n) - 1)
        # xi_ = 0.5 * (x3 + x2)-0.5 * lambda_ + p * (p / n - p / m) / (2 * (p ** 2 / (m * n) - 1))
        # delta_ = z / math.acosh(0.5 *  math.sqrt((1 + p / m) * (1 + p / n)))
        # gamma_ = delta_ * math.asinh((p / n - p / m) * math.sqrt((1 + p / m) * (1 + p / n)-4) / (2 * (p ** 2 / (m * n) - 1)))
        
        # parameters = {"xi": xi_, "lambda": lambda_, "gamma": gamma_, "delta": delta_}
        
        scipy_params = scipy.stats.johnsonsb.fit(measurements.data)
        parameters = {"xi": scipy_params[2], "lambda": scipy_params[3], "gamma": scipy_params[0], "delta": scipy_params[1]}
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
    path = "../data/data_johnson_sb.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = JOHNSON_SB(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))
