import scipy.integrate
import math
import scipy.optimize
import numpy

class KUMARASWAMY:
    """
    Kumaraswami distribution
    https://en.wikipedia.org/wiki/Kumaraswamy_distribution        
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.alpha_ = self.parameters["alpha"]
        self.beta_ = self.parameters["beta"]
        self.min_ = self.parameters["min"]
        self.max_ = self.parameters["max"]
        
    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        z = lambda t: (t - self.min_) / (self.max_ - self.min_)
        result = 1 - ( 1 - z(x) ** self.alpha_) ** self.beta_
        return result
    
    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        z = lambda t: (t - self.min_) / (self.max_ - self.min_)
        return (self.alpha_ * self.beta_) * (z(x) ** (self.alpha_ - 1)) * ((1 - z(x) ** self.alpha_) ** (self.beta_ - 1)) /  (self.max_ - self.min_)

    def get_num_parameters(self) -> int:
        """
        Number of parameters of the distribution
        """
        return len(self.parameters)
    
    def parameter_restrictions(self) -> bool:
        """
        Check parameters restrictions
        """
        v1 = self.alpha_ > 0
        v2 = self.beta_ > 0
        v3 = self.min_ < self.max_
        return v1 and v2 and v3
    
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
            ## Variables declaration
            alpha_, beta_, min_, max_ = initial_solution
            
            ## Generatred moments function (not - centered)
            E = lambda r: beta_ * math.gamma(1 + r / alpha_) * math.gamma(beta_) / math.gamma(1 + beta_ + r / alpha_)
            
            ## Parametric expected expressions
            parametric_mean = E(1) * (max_ - min_) + min_
            parametric_variance = (E(2) - E(1) ** 2) * (max_ - min_) ** 2
            parametric_skewness = (E(3) - 3 * E(2) * E(1) + 2 * E(1) ** 3) / ((E(2) - E(1) ** 2)) ** 1.5
            parametric_kurtosis = (E(4)-4 * E(1) * E(3) + 6 * E(1) ** 2 * E(2) - 3 * E(1) ** 4) /  ((E(2) - E(1) ** 2)) ** 2
            parametric_median = ((1 - 2 ** (-1 / beta_)) ** (1 / alpha_)) * (max_ - min_) + min_
            
            ## System Equations
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            # eq2 = parametric_median - measurements.median
            eq3 = parametric_skewness - measurements.skewness
            eq4 = parametric_kurtosis  - measurements.kurtosis
            
            return (eq1, eq2, eq3, eq4)
        
        # solution = scipy.optimize.fsolve(equations, (1, 1, 1, 1), measurements)
        l = measurements.min - 3 * abs(measurements.min)
        bnds = ((0, 0, l, l), (numpy.inf, numpy.inf, numpy.inf, numpy.inf))
        x0 = (1, 1, 1, 1)
        args = ([measurements])
        solution = scipy.optimize.least_squares(equations, x0, bounds = bnds, args=args)
        
        parameters = {"alpha": solution.x[0], "beta": solution.x[1], "min": solution.x[2], "max": solution.x[3]}
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
    path = "../data/data_kumaraswamy.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = KUMARASWAMY(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))
    