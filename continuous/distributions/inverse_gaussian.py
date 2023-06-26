import math
import scipy.stats

class INVERSE_GAUSSIAN:
    """
    Inverse Gaussian distribution
    Also known like Wald distribution
    https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution        
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.miu = self.parameters["miu"]
        self.lambda_ = self.parameters["lambda"]

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        # result = scipy.stats.invgauss.cdf(x, self.miu/self.lambda_, scale=self.lambda_)
        result = scipy.stats.norm.cdf(math.sqrt(self.lambda_ / x) * ((x / self.miu) - 1)) + math.exp(2 * self.lambda_ / self.miu) * scipy.stats.norm.cdf(-math.sqrt(self.lambda_ / x) * ((x / self.miu) + 1))
        return result
    
    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        # result = scipy.stats.invgauss.pdf(x, self.miu/self.lambda_, scale=self.lambda_)
        result = math.sqrt(self.lambda_ / (2 * math.pi * x ** 3)) * math.exp(-(self.lambda_ * (x - self.miu) ** 2) / (2 * self.miu ** 2 * x))
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
        v1 = self.miu > 0
        v2 = self.lambda_ > 0
        return v1 and v2

    def get_parameters(self, measurements) -> dict[str, float | int]:
        """
        Calculate proper parameters of the distribution from sample measurements.
        The parameters are calculated by formula.
        
        Parameters
        ==========
        measurements : dict
            {"miu":  * , "variance":  * , "skewness":  * , "kurtosis":  * , "data":  * }

        Returns
        =======
        parameters : dict
            {"miu":  * , "lambda":  * }
        """
        μ = measurements.mean
        λ = μ ** 3 / measurements.variance
        
        parameters = {"miu": μ, "lambda": λ}
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
    path = "../data/data_inverse_gaussian.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = INVERSE_GAUSSIAN(measurements)
    
    print(distribution.get_parameters(measurements))
    print(scipy.stats.invgauss.fit(data))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))