import math
import scipy.optimize
import numpy
import scipy.special as sc
import scipy.stats

class BETA:
    """
    Beta distribution
    https://en.wikipedia.org/wiki/Beta_distribution
    Compendium of Common Probability Distributions (pag.23) ... Michael P. McLaughlin
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.alpha_ = self.parameters["alpha"]
        self.beta_ = self.parameters["beta"]
        self.A = self.parameters["A"]
        self.B_ = self.parameters["B"]
        
    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        z = lambda t: (t - self.A) / (self.B_ - self.A)
        # result = scipy.stats.beta.cdf(z(x), self.alpha_, self.beta_)
        # result = print(result, error = scipy.integrate.quad(self.pdf, self.A, x)
        result = sc.betainc(self.alpha_, self.beta_, z(x))
        
        return result
    
    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        z = lambda t: (t - self.A) / (self.B_ - self.A)
        return ( 1 / (self.B_ - self.A)) * ( math.gamma(self.alpha_ + self.beta_) / (math.gamma(self.alpha_) * math.gamma(self.beta_))) * (z(x) ** (self.alpha_ - 1)) * ((1 - z(x)) ** (self.beta_ - 1))

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
        v3 = self.A < self.B_
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
            {"alpha":  * , "beta":  * , "A":  * , "B":  * }
        """
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            ## Variables declaration
            alpha_, beta_, A, B_ = initial_solution
            
            ## Parametric expected expressions
            parametric_mean = A + (alpha_ / ( alpha_ + beta_ )) * (B_ - A)
            parametric_variance = ((alpha_ * beta_) / ((alpha_ + beta_) ** 2 * (alpha_ + beta_ + 1))) * (B_ - A) ** 2
            parametric_skewness = 2 * ((beta_ - alpha_) / (alpha_ + beta_ + 2)) * math.sqrt((alpha_ + beta_ + 1) / (alpha_ * beta_))
            parametric_kurtosis = 3 * (((alpha_ + beta_ + 1) * (2 * (alpha_ + beta_) ** 2  + (alpha_ * beta_) * (alpha_ + beta_ - 6))) / ((alpha_ * beta_) * (alpha_ + beta_ + 2) * (alpha_ + beta_ + 3)))
            
            ## System Equations
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            eq3 = parametric_skewness - measurements.skewness
            eq4 = parametric_kurtosis  - measurements.kurtosis
            
            return (eq1, eq2, eq3, eq4)
        
        bnds = ((0, 0,  - numpy.inf, measurements.mean), (numpy.inf, numpy.inf, measurements.mean, numpy.inf))
        x0 = (1, 1, measurements.min, measurements.max)
        args = ([measurements])
        solution = scipy.optimize.least_squares(equations, x0, bounds = bnds, args=args)
        parameters = {"alpha": solution.x[0], "beta": solution.x[1], "A": solution.x[2], "B": solution.x[3]}
        
        v1 = parameters["alpha"] > 0
        v2 = parameters["beta"] > 0
        v3 = parameters["A"] < parameters["B"]
        if ((v1 and v2 and v3) == False):
            scipy_params = scipy.stats.beta.fit(measurements.data)
            parameters = {"alpha": scipy_params[0], "beta": scipy_params[1], "A": scipy_params[2], "B": scipy_params[3]}
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
    path = "../data/data_beta.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS_CONTINUOUS(data)
    distribution = BETA(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))
    
    
    
    # def equations(initial_solution: list[float], measurements) -> tuple[float]:
    #     ## Variables declaration
    #     alpha_, beta_, A, B_ = initial_solution
        
    #     ## Parametric expected expressions
    #     parametric_mean = A + (alpha_ / ( alpha_ + beta_ )) * (B_ - A)
    #     parametric_variance = ((alpha_ * beta_) / ((alpha_ + beta_) ** 2 * (alpha_ + beta_ + 1))) * (B_ - A) ** 2
    #     parametric_skewness = 2 * ((beta_ - alpha_) / (alpha_ + beta_ + 2)) * math.sqrt((alpha_ + beta_ + 1) / (alpha_ * beta_))
    #     parametric_kurtosis = 3 * (((alpha_ + beta_ + 1) * (2 * (alpha_ + beta_) ** 2  + (alpha_ * beta_) * (alpha_ + beta_ - 6))) / ((alpha_ * beta_) * (alpha_ + beta_ + 2) * (alpha_ + beta_ + 3)))
        
    #     ## System Equations
    #     eq1 = parametric_mean - measurements.mean
    #     eq2 = parametric_variance - measurements.variance
    #     eq3 = parametric_skewness - measurements.skewness
    #     eq4 = parametric_kurtosis  - measurements.kurtosis
        
    #     return (eq1, eq2, eq3, eq4)
    
    # ## Get parameters of distribution: SCIPY vs EQUATIONS
    # import time
    # print("=====")
    # ti = time.time()
    # solution = scipy.optimize.fsolve(equations, (1, 1, 1, 1), measurements)
    # parameters = {"alpha": solution[0], "beta": solution[1], "A": solution[2], "B": solution[3]}
    # print(parameters)
    # print("Solve equations time: ", time.time() - ti)
    
    # print("=====")
    # ti = time.time()
    # scipy_params = scipy.stats.beta.fit(measurements.data)
    # parameters = {"alpha": scipy_params[0], "beta": scipy_params[1], "A": scipy_params[2], "B": scipy_params[3]}
    # print(parameters)
    # print("Scipy time get parameters: ",time.time() - ti)
    
    # print("=====")
    
    # ti = time.time()
    # bnds = ((0, 0,  - numpy.inf, measurements.mean), (numpy.inf, numpy.inf, measurements.mean, numpy.inf))
    # x0 = (1, 1, measurements.min, measurements.max)
    # args = ([measurements])
    # solution = scipy.optimize.least_squares(equations, x0, bounds = bnds, args=args)
    # print(solution.x)
    # print("Solve equations time: ", time.time() - ti)
    
