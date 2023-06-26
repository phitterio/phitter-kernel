import scipy.integrate
import math
import scipy.stats
import scipy.special as sc
import numpy

class NC_F:
    """
    Non-Central F distribution
    https://en.wikipedia.org/wiki/Noncentral_F-distribution
    Hand-book on Statistical Distributions (pag.113) ... Christian Walck      
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.lambda_ = self.parameters["lambda"]
        self.n1 = self.parameters["n1"]
        self.n2 = self.parameters["n2"]
        
    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        ## Method 1
        # result = scipy.stats.ncf.cdf(x, self.n1, self.n2, self.lambda_)
        
        ## Method 2
        result = sc.ncfdtr(self.n1, self.n2, self.lambda_, x)
        
        ## Method 3
        # k = 0
        # acum = 0
        # r0 = -1
        # while(acum - r0 > 1e-10):
        #     r0 = acum
        #     t1 = ((self.lambda_ / 2) ** k) / math.factorial(k)
        #     q = x * self.n1 / (self.n2 + x * self.n1)
        #     t2 = sc.betainc(k + self.n1 / 2, self.n2 / 2, q)
        #     s = t1 * t2
        #     acum += s
        #     k += 1
        # result = math.exp(-self.lambda_ / 2) * acum
        return result
    
    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        ## Method 1
        result = scipy.stats.ncf.pdf(x, self.n1, self.n2, self.lambda_)
        
        ## Method 2
        # k = 0
        # acum = 0
        # r0 = -1
        # while(acum - r0 > 1e-10):
        #     r0 = acum
        #     t1 = 1 / math.factorial(k)
        #     t2 = (self.lambda_ / 2) ** k
        #     t3 = 1 / sc.beta(k + self.n1 / 2, self.n2 / 2)
        #     t4 = (self.n1 / self.n2) ** (k + self.n1 / 2)
        #     t5 = (self.n2 / (self.n2 + self.n1 * x)) ** (k + (self.n1 + self.n2) / 2)
        #     t6 = x ** (k - 1 + self.n1 / 2)
        #     s = t1 * t2 * t3 * t4 * t5 * t6
        #     acum += s
        #     k += 1
        # result = math.exp(-self.lambda_ / 2) * acum
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
        v1 = self.lambda_ > 0
        v2 = self.n1 > 0
        v3 = self.n2 > 0
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
            {"df":  * }
        """
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            lambda_, n1, n2 = initial_solution
            
            ## Generatred moments function (not - centered)
            E_1 = (n2 / n1) * ((n1 + lambda_) / (n2 - 2))
            E_2 = (n2 / n1) ** 2 * ((lambda_ ** 2 + (2  * lambda_ + n1) * (n1 + 2)) / ((n2 - 2) * (n2-4)))
            E_3 = (n2 / n1) ** 3 * ((lambda_ ** 3 + 3 * (n1 + 4) * lambda_ ** 2 + (3 * lambda_ + n1) * (n1 + 2) * (n1 + 4)) /  ((n2 - 2) * (n2-4) * (n2 - 6)))
            # E_4 = (n2 / n1) ** 4 * ((lambda_ ** 4 + 4 * (n1 + 6) * lambda_ ** 3 + 6 * (n1 + 4) * (n1 + 6) * lambda_ ** 2 + (4 * lambda_ + n1) * (n1 + 2) * (n1 + 4) * (n1 + 4)) /  ((n2 - 2) * (n2-4) * (n2 - 6) * (n2 - 6)))
            
            ## Parametric expected expressions
            parametric_mean = E_1
            parametric_variance = E_2 - E_1 ** 2
            parametric_skewness = (E_3 - 3 * E_2 * E_1 + 2 * E_1 ** 3) / ((E_2 - E_1 ** 2)) ** 1.5
            # parametric_kurtosis = (E_4-4 * E_1 * E_3 + 6 * E_1 ** 2 * E_2 - 3 * E_1 ** 4) /  ((E_2 - E_1 ** 2)) ** 2
            
            ## System Equations
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            eq3 = parametric_skewness - measurements.skewness
            # eq4 = parametric_kurtosis  - measurements.kurtosis
            
            return (eq1, eq2, eq3)
        
        bnds = ((0, 0, 0), (numpy.inf, numpy.inf, numpy.inf))
        x0 = (measurements.mean, 1, 10)
        args = ([measurements])
        solution = scipy.optimize.least_squares(equations, x0, bounds = bnds, args=args)
        parameters = {"lambda": solution.x[0], "n1": solution.x[1], "n2": solution.x[2]}
        
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
    path = "../data/data_nc_f.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = NC_F(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))