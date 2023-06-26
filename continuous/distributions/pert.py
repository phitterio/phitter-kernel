import numpy
import scipy.optimize
import scipy.special as sc


class PERT:
    """
    Pert distribution
    https://en.wikipedia.org/wiki/PERT_distribution    
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]
        self.c = self.parameters["c"]
        
    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        α1 = (4 * self.b + self.c - 5 * self.a) / (self.c - self.a)
        α2 = (5 * self.c - self.a-4 * self.b) / (self.c - self.a)
        z = lambda t: (t - self.a) / (self.c - self.a)
        
        # result = scipy.stats.beta.cdf(z(x), α1, α2)
        result = sc.betainc(α1, α2, z(x))
        
        return result
    
    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        α1 = (4 * self.b + self.c - 5 * self.a) / (self.c - self.a)
        α2 = (5 * self.c - self.a-4 * self.b) / (self.c - self.a)
        return (x - self.a) ** (α1 - 1) * (self.c - x) ** (α2 - 1) / (sc.beta(α1, α2) * (self.c - self.a) ** (α1 + α2 - 1))

    def get_num_parameters(self) -> int:
        """
        Number of parameters of the distribution
        """
        return len(self.parameters)
    
    def parameter_restrictions(self) -> bool:
        """
        Check parameters restrictions
        """
        v1 = self.a < self.b
        v2 = self.b < self.c
        return v1 and v2
    
    def get_parameters(self, measurements) -> dict[str, float | int]:
        """
        Calculate proper parameters of the distribution from sample measurements.
        The parameters are calculated by solving the equations of the measures expected 
        for this distribution.The number of equations to consider is equal to the number 
        of parameters.
        
        Parameters
        ==========
        measurements : dict
            {"mean":  * , "variance":  * , "skewness":  * , "kurtosis":  * , "median":  * , "b":  * }

        Returns
        =======
        parameters : dict
            {"a":  * , "b":  * , "c":  * }
        """
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            a, b, c = initial_solution
        
            α1 = (4 * b + c - 5 * a) / (c - a)
            α2 = (5 * c - a-4 * b) / (c - a)
            
            parametric_mean = (a + 4 * b + c) / 6
            parametric_variance = ((parametric_mean - a) * (c - parametric_mean)) / 7
            # parametric_skewness = 2 * (α2 - α1) * math.sqrt(α2 + α1 + 1) / ((α2 + α1 + 2) *  math.sqrt(α2 * α1))
            # parametric_kurtosis = 3 + 6 * ((α2 - α1) ** 2 * (α2 + α1 + 1) - (α2 * α1) * (α2 + α1 + 2)) / ((α2 * α1) * (α2 + α1 + 2) * (α2 + α1 + 3))
            # parametric_median = (a + 6 * b + c) / 8
            parametric_median = sc.betaincinv(α1, α2, 0.5) * (c - a) + a
            
            ## System Equations
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            # eq3 = parametric_skewness - measurements.skewness
            # eq4 = parametric_kurtosis  - measurements.kurtosis
            eq5 = parametric_median  - measurements.median
            
            return (eq1, eq2, eq5)
        
        ## Parameters of equations system
        bnds = (
            (-numpy.inf, measurements.mean, measurements.min),
            (measurements.mean, numpy.inf, measurements.max)
        )
        x0 = (measurements.min, measurements.mean, measurements.max)
        args = ([measurements])
        
        ## Solve Equation system
        solution = scipy.optimize.least_squares(equations, x0, bounds = bnds, args=args)
        parameters = {"a": solution.x[0], "b": solution.x[1], "c": solution.x[2]}
        
        ## Correction of parameters
        parameters["a"] = min(measurements.min - 1e-3, parameters["a"])
        parameters["c"] = max(measurements.max + 1e-3, parameters["c"])
        
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
    path = "../data/data_pert.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = PERT(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))
    




    def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
        a, c, b = initial_solution
    
        # α1 = (4 * b + c - 5 * a) / (c - a)
        # α2 = (5 * c - a-4 * b) / (c - a)
        
        parametric_mean = (a + 4 * b + c) / 6
        parametric_variance = ((parametric_mean - a) * (c - parametric_mean)) / 7
        # parametric_skewness = 2 * (α2 - α1) * math.sqrt(α2 + α1 + 1) / ((α2 + α1 + 2) *  math.sqrt(α2 * α1))
        # parametric_kurtosis = 3 + 6 * ((α2 - α1) ** 2 * (α2 + α1 + 1) - (α2 * α1) * (α2 + α1 + 2)) / ((α2 * α1) * (α2 + α1 + 2) * (α2 + α1 + 3))
        parametric_median = (a + 6 * b + c) / 8
        
        ## System Equations
        eq1 = parametric_mean - measurements.mean
        eq2 = parametric_variance - measurements.variance
        # eq3 = parametric_skewness - measurements.skewness
        # eq4 = parametric_kurtosis  - measurements.kurtosis
        eq5 = parametric_median  - measurements.median
        
        return (eq1, eq2, eq5)
    
    import time
    print("=====")
    ti = time.time()
    bnds = ((-numpy.inf, measurements.mean, measurements.min), (measurements.mean, numpy.inf, measurements.max))
    x0 = (measurements.min, measurements.max, measurements.mean)
    args = ([measurements])
    solution = scipy.optimize.least_squares(equations, x0, bounds = bnds, args=args)
    parameters = {"min": solution.x[0], "max": solution.x[1], "b": solution.x[2]}
    print(parameters)
    print("Solve equations time: ", time.time() - ti)