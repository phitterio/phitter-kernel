import numpy
import scipy.stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

mpl.style.use("ggplot")

def num_bins_doane(data):
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

def plot_histogram(data, distribution):
    plt.figure(figsize=(8, 4))
    plt.hist(data, density=True, ec='white', bins=num_bins_doane(data))
    plt.title('HISTOGRAM')
    plt.xlabel('Values')
    plt.ylabel('Frequencies')
    
    
    x_plot = numpy.linspace(min(data), max(data), 1000)
    y_plot = distribution.pdf(x_plot)
    plt.plot(x_plot, y_plot, label="PDF KDE")    
    
    plt.legend(title='DISTRIBUTIONS', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    
def getData(direction):
    sample_distribution_file = open(direction, "r")
    data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
    return data

if __name__ == "__main__":
    ## Get Data
    path = "../data/data_normal.txt"
    data = getData(path)
    
    ## KDE
    distribution = scipy.stats.gaussian_kde(data)
    
    ## Plot result
    plot_histogram(data, distribution)
    
    ## Prove PDF distribution
    print(distribution.pdf(50000))
    print(distribution.evaluate(50000))
    