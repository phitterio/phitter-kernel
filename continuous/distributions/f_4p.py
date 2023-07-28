import scipy.stats
import scipy.special as sc
import math
import numpy

class F_4P:
    """
    F distribution
    https://en.wikipedia.org/wiki/F-distribution
    http://atomic.phys.uni-sofia.bg/local/nist-e-handbook/e-handbook/eda/section3/eda366a.htm
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.df1 = self.parameters["df1"]
        self.df2 = self.parameters["df2"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]
        
    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        z = lambda t: (t - self.loc) / self.scale
        # print(scipy.stats.f.cdf(z(x), self.df1, self.df2))
        return sc.betainc(self.df1 / 2, self.df2 / 2, z(x) * self.df1 / (self.df1 * z(x) + self.df2))
    
    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        z = lambda t: (t - self.loc) / self.scale
        # print(scipy.stats.f.pdf(x, self.df1, self.df2, self.loc, self.scale))
        return (1 / self.scale) * (1 / sc.beta(self.df1 / 2, self.df2 / 2)) * ((self.df1 / self.df2) ** (self.df1 / 2)) * (z(x) ** (self.df1 / 2 - 1)) * ((1 + z(x) * self.df1 / self.df2) ** (-1 * (self.df1 + self.df2) / 2))
    
    def get_num_parameters(self) -> int:
        """
        Number of parameters of the distribution
        """
        return len(self.parameters)
    
    def parameter_restrictions(self) -> bool:
        """
        Check parameters restrictions
        """
        v1 = self.df1 > 0
        v2 = self.df2 > 0
        v3 = self.scale > 0
        return v1 and v2 and v3
        
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
            {"df1":  * , "df2":  * }
        """
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            ## Variables declaration
            df1, df2, loc, scale = initial_solution
            
            ## Generatred moments function (not - centered)
            E = lambda k: (df2 / df1) ** k * (math.gamma(df1 / 2 + k) * math.gamma(df2 / 2 - k))  / (math.gamma(df1 / 2) * math.gamma(df2 / 2))
            
            ## Parametric expected expressions
            parametric_mean = E(1) * scale + loc
            parametric_variance = (E(2) - E(1) ** 2) * (scale) ** 2
            # parametric_skewness = (E(3) - 3 * E(2) * E(1) + 2 * E(1) ** 3) / ((E(2) - E(1) ** 2)) ** 1.5
            # parametric_kurtosis = (E(4)-4 * E(1) * E(3) + 6 * E(1) ** 2 * E(2) - 3 * E(1) ** 4) /  ((E(2) - E(1) ** 2)) ** 2
            parametric_median = scipy.stats.f.ppf(0.5, df1, df2) * scale + loc
            parametric_mode = ((df2 * (df1 - 2)) / (df1 * (df2 + 2))) * scale + loc
            
            ## System Equations
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            # eq2 = parametric_median - measurements.median
            # eq2 = parametric_skewness - measurements.skewness
            # eq2 = parametric_kurtosis  - measurements.kurtosis
            eq3 = parametric_median - measurements.median
            eq4 = parametric_mode - measurements.mode
            return (eq1, eq2, eq3, eq4)
        
        try:
            bnds = ((0, 0,  - numpy.inf, 0), (numpy.inf, numpy.inf, measurements.min, numpy.inf))
            x0 = (1, measurements.standard_deviation, measurements.min, measurements.standard_deviation)
            args = ([measurements])
            solution = scipy.optimize.least_squares(equations, x0, bounds = bnds, args=args)
            parameters = {"df1": solution.x[0], "df2": solution.x[1], "loc": solution.x[2], "scale": solution.x[3]}
        except:
            ## Scipy parameters of distribution
            scipy_params = scipy.stats.f.fit(measurements.data)
           
            ## Results
            parameters = {"df1": scipy_params[0], "df2": scipy_params[1], "loc": scipy_params[2], "scale": scipy_params[3]}

        return parameters
    
if __name__ == "__main__":
    ## Import function to get measurements
    import sys
    sys.path.append("../measurements")
    from measurements import MEASUREMENTS_CONTINUOUS

    ## Import function to get measurements
    def get_data(direction):
        sample_distribution_file = open(direction, "r")
        data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
        return data
    
    ## Distribution class
    path = "../data/data_f_4p.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS_CONTINUOUS(data)
    distribution = F_4P(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))