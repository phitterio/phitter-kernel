import scipy.optimize
import numpy
import math
import scipy.stats

class POWER_FUNCTION:
    """
    Power function distribution
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.alpha = self.parameters["alpha"]
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]
        
    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        return ((x - self.a) / (self.b - self.a)) ** self.alpha
    
    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        return self.alpha * ((x - self.a) ** (self.alpha - 1)) / ((self.b - self.a) ** self.alpha)

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
        v2 = self.b > self.a
        return v1 and v2
    
    def get_parameters(self, measurements) -> dict[str, float | int]:
        """
        Calculate proper parameters of the distribution from sample measurements.
        The parameters are calculated by solving the equations of the measures expected 
        for this distribution.The number of equations to consider is equal to the number 
        of parameters.
        
        Parameters
        ==========
        measurements: MEASUREMESTS
            attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, length, num_bins, data

        Returns
        =======
        parameters : dict
            {"alpha":  * , "beta":  * , "min":  * , "max":  * }
        """
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            α, a, b= initial_solution
            
            E1 = (a + b * α) / (1 + α)
            E2 = (2 * a ** 2 + 2 * a * b * α + b ** 2 * α * (1 + α)) / ((1 + α) * (2 + α))
            E3 = (6 * a ** 3 + 6 * a ** 2 * b * α + 3 * a * b ** 2 * α * (1 + α) + b ** 3 * α * (1 + α) * (2 + α)) / ((1 + α) * (2 + α) * (3 + α))
            # E4 = (24 * a ** 4 + 24 * α * a ** 3 * b + 12 * α * (α + 1) * a ** 2 * b ** 2 + 4 * α * (α + 1) * (α + 2) * a * b ** 3 + α * (α + 1) * (α + 2) * (α + 3) * b ** 4) / ((α + 1) * (α + 2) * (α + 3) * (α + 4))
            
            parametric_mean = E1
            parametric_variance = (E2 - E1 ** 2)
            parametric_skewness = (E3 - 3 * E2 * E1 + 2 * E1 ** 3) / ((E2 - E1 ** 2)) ** 1.5
            # parametric_kurtosis = (E4-4 * E1 * E3 + 6 * E1 ** 2 * E2 - 3 * E1 ** 4) /  ((E2 - E1 ** 2)) ** 2
            # parametric_median = (0.5 ** (1 / α)) * (b - a) + a
        
            ## System Equations
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            eq3 = parametric_skewness - measurements.skewness
            # eq4 = parametric_kurtosis  - measurements.kurtosis
            # eq5 = parametric_median - measurements.median
            
            return (eq1, eq2, eq3)

        bnds = ((0,  - numpy.inf,  - numpy.inf), (numpy.inf, numpy.inf, numpy.inf))
        x0 = (1, 1, measurements.max)
        args = ([measurements])
        solution = scipy.optimize.least_squares(equations, x0, bounds = bnds, args=args)
        parameters = {"alpha": solution.x[0], "a": solution.x[1], "b": measurements.max + 1e-3}
        
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
    path = "../data/data_power_function.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS_CONTINUOUS(data)
    distribution = POWER_FUNCTION(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))
        
    print(scipy.stats.powerlaw.fit(measurements.data))