from scipy import optimize
import scipy.integrate


def derivative(f, a, h=0.001):
    """
    Approximates the derivative of the function f in a
    
    Parameters
    ==========
    f: function ...  function to differentiate
    a: float ... the point of differentiation
    h: float ... step size
    
    Return
    ======
    result: float: the derivative of f in a
    """
    return (f(a + h) - f(a - h)) / (2 * h)
  
def riemann_stieltjes_integral(f, g, a, b, n):
    """
    Calculates the Riemann - Stieltjes integral based on the composite trapezoidal rule
    relying on the Riemann Sums. integral(f(x)dg(x))
    
    Parameters
    ==========
    f: function ... integrand function
    g: function ... integrator function 
    a: int ... lower bound of the integral
    b: int ... upper bound of theintergal
    n: int ... number of trapezoids of equal width
    
    Return
    ======
    result: float ... the integral of the function f between a and b
    
    https://gist.github.com / IlievskiV / 8004a94dd508d723678d8da57002f157#file - riemann_stieltjes_integration - py
    """
    eps = 1e-9
    h = (b - a) / (n + eps)  # width of the rectangle
    dg = lambda x: derivative(g, x)  # derivative of the integrator function
    result = 0.5 * f(a) * dg(a) + sum([f(a + i * h) * dg(a + i * h) for i in range(1, n)]) + 0.5 * f(b) * dg(b)
    result *= h
    
    return result

def empirical_distribution(x, data):
    cont = 0
    for d in data:
        if d <= x:
            cont += 1
    return cont / len(data)

def inv(data, distribution, parameters, value, x0):
    f = lambda x: distribution.cdf(x, parameters) - value
    root = optimize.newton(f, x0)
    return root

def anderson_darling_distribution(data, distribution, parameters):
    Fn = lambda x: empirical_distribution(x, data)
    F = lambda x: distribution.cdf(x, parameters)
    
    G = lambda x:(( Fn(x) - F(x)) ** 2) / (F(x) * (1 - F(x)))
    
    # acum = 0
    # for i in range(len(data)):
    #     # print(empirical_distribution(data[i], data), distribution.cdf(data[i], parameters))
    #     acum += abs(empirical_distribution(data[i], data) - distribution.cdf(data[i], parameters)) ** 2
    # print(acum)
    
    inf = inv(data, distribution, parameters, 1e-8, min(data))
    sup = inv(data, distribution, parameters, 1 - 1e-8, max(data))
                
    I = riemann_stieltjes_integral(G, F, inf, sup, 100)
        
    # I = riemann_stieltjes_integral(f, g, min(data), max(data), 200)
    return len(data) * I
    
def ad_distribution(data, distribution, parameters):
    Fn = lambda x: empirical_distribution(x, data)
    F = lambda x: distribution.cdf(x, parameters)
    f = lambda x: distribution.pdf(x, parameters)
    
    G = lambda x:  f(x) * (( Fn(x) - F(x)) ** 2) / (F(x) * (1 - F(x)))
    
    inf = min(data)
    sup = max(data)
    
    # print(scipy.integrate.quad(G, inf - 10, inf))
    # print(scipy.integrate.quad(G, sup, sup + 10))
    
    I = scipy.integrate.quad(G, inf, sup, limit=100, full_output=True)
    return I[0] * len(data)
    
if __name__ == "__main__":
    from data_measurements import get_measurements
    import beta
    import chi_square
    import exponencial
    import gamma
    import johnson_SB
    import lognormal
    import normal
    import weibull
    
    def getData(direction):
        sample_distribution_file = open(direction, "r")
        data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
        return data
    
    _all_distributions = [beta, chi_square, exponencial, gamma, johnson_SB, lognormal, normal, weibull]
    distribution = normal
    
    path = path = "C: / Users / USUARIO1 / Desktop / Fitter / data / data_normal.txt"
    data = getData(path)
    
    measurements = MEASUREMENTS(data)
    parameters = distribution.get_parameters(measurements)
    print(parameters)
    
    print(anderson_darling_distribution(data, distribution, parameters))
    print(ad_distribution(data, distribution, parameters))
    
    # for distribution in _all_distributions:
    #     print(str(distribution.__file__)[33: - 3])
    #     path = "C: / Users / USUARIO1 / Desktop / Fitter / data / data_" + str(distribution.__file__)[33: - 3]  + ".txt"
    #     data = getData(path)
        
    #     measurements = MEASUREMENTS(data)
    #     parameters = distribution.get_parameters(measurements)
    #     print(parameters)
        
    #     anderson_darling_distribution(data, distribution, parameters)
    
    
    