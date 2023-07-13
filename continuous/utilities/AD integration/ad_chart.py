import numpy
import matplotlib.pyplot as plt
import math

def empirical_distribution(x, data):
    cont = 0
    for d in data:
        if d <= x:
            cont += 1
    return cont / len(data)

def graph(data, distribution, parameters):
    Fn = lambda x: empirical_distribution(x, data)
    F = lambda x: distribution.cdf(x, parameters)
    f = lambda x: distribution.pdf(x, parameters)
    
    G = lambda x:  f(x) * (( Fn(x) - F(x)) ** 2) / (F(x) * (1 - F(x)))
    
    
    x = numpy.arange(min(data), max(data), 0.1)
    y = numpy.array([G(x0) for x0 in x])
    
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # potting the points
    plt.plot(x, y, linewidth=1.5, color='firebrick') 
    plt.axvline(x=min(data), linestyle=' -  - ', linewidth=1.5, color='royalblue')
    plt.axvline(x=max(data), linestyle=' -  - ', linewidth=1.5, color='royalblue')
    
    # function to show the plot
    ax.set_title('Anderson Darling Distribution')
    ax.set_xlabel('x')
    ax.set_ylabel('G(x)')
    # ax.set_ylim([min(y) - 5,max(y) + 20])
    plt.show()
    
if __name__ == "__main__":
    import sys
    sys.path.append("../measurements")
    from measurements import MEASUREMENTS
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
    
    path = "C: / Users / USUARIO1 / Desktop / Fitter / data / data_normal.txt"
    data = getData(path)
    
    measurements = MEASUREMENTS(data)
    parameters = distribution.get_parameters(measurements)
    print(parameters)
    
    graph(data, distribution, parameters)
    
    # for distribution in _all_distributions:
    #     print(str(distribution.__file__)[33: - 3])
    #     path = "C: / Users / USUARIO1 / Desktop / Fitter / data / data_" + str(distribution.__file__)[33: - 3]  + ".txt"
    #     data = getData(path)
        
    #     measurements = MEASUREMENTS(data)
    #     parameters = distribution.get_parameters(measurements)
    #     print(parameters)
        
    #     anderson_darling_distribution(data, distribution, parameters)