import scipy.stats
import math
import numpy
import scipy.special as sc
import scipy.integrate

class NON_CENTRAL_T_STUDENT:
    """
    Non-Central T Student distribution
    https://en.wikipedia.org/wiki/Noncentral_t-distribution
    Hand-book on Statistical Distributions (pag.116) ... Christian Walck      
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.lambda_ = self.parameters["lambda"]
        self.n = self.parameters["n"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]
        
    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        ##  Method 1
        z = lambda x: (x - self.loc) / self.scale
        result = sc.nctdtr(self.n, self.lambda_, z(x))
        
        ## Method 2
        # result = scipy.stats.nct.cdf(z(x), self.n, self.lambda_)
        
        ## Method 3
        # k = 0
        # acum = 0
        # r0 = -1
        # while(acum - r0 > 1e-20):
        #     r0 = acum
        #     t1 = math.exp(-self.lambda_ ** 2 / 2) * (self.lambda_ ** 2 / 2) ** k / math.factorial(k)
        #     t2 = math.exp(-self.lambda_ ** 2 / 2) * (self.lambda_ ** 2 / 2) ** k * self.lambda_ / (math.sqrt(2) * math.gamma(k + 1.5))
        #     y = (z(x) ** 2) / (z(x) ** 2 + self.n)
        #     s = t1 * sc.betainc(k + 0.5, self.n / 2, y) + t2 * sc.betainc(k + 1, self.n / 2, y)
        #     acum += s
        #     k += 1
        # result = scipy.stats.norm.cdf(-self.lambda_) + 0.5 * acum
        # print(result)        
        
        ## Method 4
        # result, err = scipy.integrate.quad(self.pdf,  - numpy.inf, x)
        return result
    
    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        ## Method 1
        z = lambda x: (x - self.loc) / self.scale
        result = scipy.stats.nct.pdf(z(x), self.n, self.lambda_) / self.scale
        
        ## Method 2
        # t1 = self.n ** (self.n / 2) * math.gamma(self.n + 1)
        # t2 = 2 ** self.n * math.exp(self.lambda_ ** 2 / 2) * (self.n + z(x) ** 2) ** (self.n / 2) * math.gamma(self.n / 2)
        # t3 = math.sqrt(2) * self.lambda_ * z(x) * sc.hyp1f1(1 + self.n / 2, 1.5, (self.lambda_ ** 2 * z(x) ** 2) / (2 * (self.n + z(x) ** 2)))
        # t4 = (self.n + z(x) ** 2) * math.gamma(0.5 * (self.n + 1))
        # t5 = sc.hyp1f1((1 + self.n) / 2, 0.5, (self.lambda_ ** 2 * z(x) ** 2) / (2 * (self.n + z(x) ** 2)))
        # t6 = math.sqrt(self.n + z(x) ** 2) * math.gamma(1 + self.n / 2)
        # result = (t1 / t2 * (t3 / t4 + t5 / t6)) / self.scale

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
        v1 = self.n > 0
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
            {"lambda": *, "n": *, "loc": *, "scale": *}
        """
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            lambda_, n, loc, scale = initial_solution
            
            ## Generatred moments function (not - centered)
            E_1 = lambda_ * math.sqrt(n / 2) * math.gamma((n - 1) / 2) / math.gamma(n / 2)
            E_2 = (1 + lambda_ ** 2) * n / (n - 2)
            E_3 = lambda_ * (3 + lambda_ ** 2) * n ** 1.5 * math.sqrt(2) * math.gamma((n - 3) / 2) / (4 * math.gamma(n / 2))
            E_4 = (lambda_ ** 4 + 6 * lambda_ ** 2 + 3) * n ** 2 / ((n - 2) * (n-4))
            
            ## Parametric expected expressions
            parametric_mean = E_1 * scale + loc
            parametric_variance = (E_2 - E_1 ** 2) * (scale ** 2)
            parametric_skewness = (E_3 - 3 * E_2 * E_1 + 2 * E_1 ** 3) / ((E_2 - E_1 ** 2)) ** 1.5
            parametric_kurtosis = (E_4-4 * E_1 * E_3 + 6 * E_1 ** 2 * E_2 - 3 * E_1 ** 4) /  ((E_2 - E_1 ** 2)) ** 2
            
            ## System Equations
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            eq3 = parametric_skewness - measurements.skewness
            eq4 = parametric_kurtosis  - measurements.kurtosis
            
            return (eq1, eq2, eq3, eq4)
        
        bnds = ((0, 0, 0, 0), (numpy.inf, numpy.inf, numpy.inf, numpy.inf))
        x0 = (1, 5, measurements.mean, 1)
        args = ([measurements])
        solution = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
        parameters = {"lambda": solution.x[0], "n": solution.x[1], "loc": solution.x[2], "scale": solution.x[3]}
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

    path = "../data/data_NON_CENTRAL_T_STUDENT.txt"
    
    ## Distribution class
    path = "../data/data_NON_CENTRAL_T_STUDENT.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS_CONTINUOUS(data)
    distribution = NON_CENTRAL_T_STUDENT(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))
    
    
    # print(scipy.stats.nct.fit(data))