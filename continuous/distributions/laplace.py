import math
import numpy

class LAPLACE:
    """
    Laplace distribution
    https://en.wikipedia.org/wiki/Laplace_distribution 
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.miu = self.parameters["miu"]
        self.b = self.parameters["b"]
        
    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        return 0.5 + 0.5 * numpy.sign(x - self.miu) * (1 - math.exp(-abs(x - self.miu) / self.b))
    
    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        return (1 / (2 * self.b)) * math.exp(-abs(x - self.miu) / self.b)
    
    def get_num_parameters(self) -> int:
        """
        Number of parameters of the distribution
        """
        return len(self.parameters)
    
    def parameter_restrictions(self) -> bool:
        """
        Check parameters restrictions
        """
        v1 = self.b > 0
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
            {"miu":  * , "b":  * }
        """
        miu = measurements.mean
        b = math.sqrt(measurements.variance / 2)
    
        ## Results
        parameters = {"miu": miu, "b": b}

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
    path = "../data/data_laplace.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = LAPLACE(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))