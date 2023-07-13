import math
import scipy.optimize

class BRADFORD:
    """
    Bradford distribution
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.c = self.parameters["c"]
        self.min_ = self.parameters["min"]
        self.max_ = self.parameters["max"]
        
    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        result = math.log(1 + self.c * (x - self.min_) / (self.max_ - self.min_)) / math.log(self.c + 1)
        return result
    
    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        result = self.c  / ((self.c * (x - self.min_) + self.max_ - self.min_) * math.log(self.c + 1))
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
        v1 = self.max_ > self.min_
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
            {"min":  * , "max":  * }
        """
        
        _min = measurements.min - 1e-3
        _max = measurements.max + 1e-3
        
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            ## Variables declaration
            c = initial_solution
            
            ## Parametric expected expressions
            parametric_mean = (c * (_max - _min) + math.log(c + 1) * (_min * (c + 1) - _max)) / (c * math.log(c + 1))
            
            ## System Equations
            eq1 = parametric_mean - measurements.mean
            
            return (eq1)
        
        solution = scipy.optimize.fsolve(equations, (1), measurements)
        
        parameters = {"c": solution[0],"min": _min , "max": _max}
        
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

    path = "../data/data_bradford.txt"

    ## Distribution class
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = BRADFORD(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))