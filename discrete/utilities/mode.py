import numpy
import scipy.stats
import scipy.optimize
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import random

mpl.style.use("ggplot")

def doanes_formula(data):
    """
    DONAE'S FORMULA
    https://en.wikipedia.org/wiki/Histogram#Doane's_formula
    """
    N = len(data)
    skewness = scipy.stats.skew(data)
    sigma_g1 = math.sqrt((6 * (N - 2)) / ((N + 1) * (N + 3)))
    num_bins = 1 + math.log(N,2) + math.log(1 + abs(skewness) / sigma_g1,2)
    num_bins = round(num_bins)
    return num_bins

def plot_histogram(data, distribution, modes):
    plt.figure(figsize=(8, 4))
    plt.hist(data, density=True, ec='white', bins=doanes_formula(data))
    plt.title('HISTOGRAM')
    plt.xlabel('Values')
    plt.ylabel('Frequencies')
    
    
    x_plot = numpy.linspace(min(data), max(data), 1000)
    y_plot = distribution.pdf(x_plot)
    plt.plot(x_plot, y_plot, linewidth=4, label="PDF KDE")
    
    for name, mode in modes.items():
        plt.axvline(mode, linewidth=2, label=name + ": " + str(mode)[:7], color=(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
    
    plt.legend(title='DISTRIBUTIONS', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    
def getData(direction):
    sample_distribution_file = open(direction, "r")
    data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
    return data

def calc_scipy_mode(data):
    return scipy.stats.mode(data)[0]

def calc_minimize_mode(data, distribution):
    def objective(x):
        return 1 / distribution.pdf(x)[0]
    
    bnds = [(min(data), max(data))]
    solution = scipy.optimize.minimize(objective, [1], bounds = bnds)
    
    return solution.x[0]

def calc_shgo_mode(data, distribution):
    def objective(x):
        return 1 / distribution.pdf(x)[0]
    
    bnds = [[min(data), max(data)]]
    solution = scipy.optimize.shgo(objective, bounds= bnds, n=100 * len(data))
    return solution.x[0]


def calc_max_pdf_mode(data, distribution):
    x_domain = numpy.linspace(min(data), max(data), 1000)
    y_pdf = distribution.pdf(x_domain)
    i = numpy.argmax(y_pdf)
    return x_domain[i]

def calculate_mode(data):
    ## KDE
    distribution = scipy.stats.gaussian_kde(data)
    
    # scipy_mode = calc_scipy_mode(data)[0]
    # minimize_mode = calc_minimize_mode(data, distribution)
    # max_pdf_mode = calc_max_pdf_mode(data, distribution)
    shgo_mode = calc_shgo_mode(data, distribution)
    
    # modes = {
    #     "scipy_mode": scipy_mode, 
    #     "minimize_mode": minimize_mode, 
    #     "max_pdf_mode": max_pdf_mode,
    #     "shgo_mode": shgo_mode
    # }
    # plot_histogram(data, distribution, modes)
    
    return(shgo_mode)
    
if __name__ == "__main__":
    ## Get Data
    path = "../data/data_dagum_4P.txt"
    data = getData(path)
    
    calculate_mode(data)
    
    
