import scipy.optimize
import numpy
import scipy.stats
import scipy.special as sc

import warnings

warnings.filterwarnings("ignore")

class BURR_4P:
    """
    Burr distribution
    Conpendium.pdf pg.27
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.A = self.parameters["A"]
        self.B = self.parameters["B"]
        self.C = self.parameters["C"]
        self.loc = self.parameters["loc"]
        
    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        result = 1 - ((1 + ((x - self.loc) / self.A) ** (self.B )) ** (-self.C))
        return result
      
    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        result = ((self.B * self.C) / self.A) * (((x - self.loc) / self.A) ** (self.B - 1)) * ((1 + ((x - self.loc) / self.A) ** (self.B)) ** (-self.C - 1))
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
        v1 = self.A > 0
        v2 = self.C > 0
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
            {"A": * , "B":  * , "C":  * }
        """
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            ## Variables declaration
            A, B, C, loc = initial_solution
            
            ## Moments Burr Distribution
            miu = lambda r: (A ** r) * C * sc.beta((B * C - r) / B, (B + r) / B)
            
            ## Parametric expected expressions
            parametric_mean = miu(1) + loc
            parametric_variance = -(miu(1) ** 2) + miu(2)
            # parametric_skewness = 2 * miu(1) ** 3 - 3 * miu(1) * miu(2) + miu(3)
            parametric_kurtosis = -3 * miu(1) ** 4 + 6 * miu(1) ** 2 * miu(2) -4 * miu(1) * miu(3) + miu(4)
            # parametric_median = A * ((2 ** (1 / C)) - 1) ** (1 / B) + loc
            parametric_mode = A * ((B - 1) / (B * C + 1)) ** (1 / B) + loc
            
            ## System Equations
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            eq3 = parametric_kurtosis - measurements.kurtosis
            eq4 = parametric_mode - measurements.mode
            
            return (eq1, eq2, eq3, eq4)
        
        # ## Solve equations system
        # x0 = [measurements.mean, measurements.mean, measurements.mean, measurements.mean]
        # b = ((1e-5, 1e-5, 1e-5,  - numpy.inf), (numpy.inf, numpy.inf, numpy.inf, numpy.inf))
        # solution = scipy.optimize.least_squares(equations, x0, bounds = b, args=([measurements]))
        # parameters = {"A": solution.x[0], "B": solution.x[1], "C": solution.x[2], "loc": solution.x[3]}
        # print(parameters)
        
        ## Scipy class
        scipy_params = scipy.stats.burr12.fit(measurements.data)
        parameters = {"A": scipy_params[3], "B": scipy_params[0], "C": scipy_params[1], "loc": scipy_params[2]}
    
        
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
    path = "../data/data_burr_4P.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = BURR_4P(measurements)
    print(distribution.get_parameters(measurements))

    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))
