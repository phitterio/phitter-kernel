import numpy
import matplotlib as mpl
import matplotlib.pyplot as plt
from measurements.measurements import MEASUREMENTS

mpl.style.use("ggplot")

def plot_histogram(distribution, frequencies):
    ## Histogram of data
    plt.figure(figsize=(8, 4))
    plt.title('HISTOGRAM')
    plt.xlabel('Values')
    plt.ylabel('Frequencies')
    
    domain = list(frequencies.keys())
    observed = list(frequencies.values())
    expected = [distribution.pdf(d) for d in domain]

    plt.bar(domain, observed)
    plt.plot(domain, expected, color = 'royalblue', linewidth=3)
    plt.show()
        
def main():
    from distributions.bernoulli import BERNOULLI
    from distributions.binomial import BINOMIAL
    from distributions.geometric import GEOMETRIC
    from distributions.hypergeometric import HYPERGEOMETRIC
    from distributions.logarithmic import LOGARITHMIC
    from distributions.negative_binomial import NEGATIVE_BINOMIAL
    from distributions.poisson import POISSON
    from distributions.uniform import UNIFORM
    
    def get_data(direction):
        sample_distribution_file = open(direction, "r")
        data = [int(x) for x in sample_distribution_file.read().splitlines()]
        return data
    
    distribution_class = POISSON
    path = "./data/data_" + distribution_class.__name__.lower() + ".txt"
    ## path = "./data/data_poisson.txt"
    data = get_data(path)
    measurements = MEASUREMENTS(data)
    frequencies = measurements.frequencies
    frequencies = {k: v / measurements.length for k,v in frequencies.items() }
    distribution = distribution_class(measurements)

    plot_histogram(distribution, frequencies)

if __name__ == "__main__":
    main()
    