import numpy
import matplotlib.pyplot as plt

def graph(data, distribution, parameters):
   
    numpy.random.seed(19680801)
    
    n_bins = 50
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # plot the cumulative histogram
    n, bins, patches = ax.hist(data,  n_bins, density=True, histtype='step', color='steelblue',
                               cumulative=True, linewidth=1.5, label='Empirical')
    
    # Add a line showing the expected distribution.
    y = numpy.array([distribution.pdf(x, parameters) for x in bins])
    y = y.cumsum()
    y /= y[ - 1]
    plt.style.use('Solarize_Light2')
    
    ax.plot(bins, y, 'k -  - ', color='firebrick', linewidth=1.5, label=str(distribution.__file__)[33: - 3])
    
    # Overlay a reversed cumulative histogram.
    # ax.hist(data, bins=bins,density=True, histtype='step', cumulative= - 1, 
    #         color='slategrey', linewidth=1.5, label='Reversed emp.')
    
    # tidy up the figure
    ax.grid(True)
    ax.legend(loc=2)
    ax.set_title('Empirical vs CDF')
    ax.set_xlabel('Range')
    ax.set_ylabel('Likelihood of occurrence')
    ax.set_ylim([-0.03,1.03])

    plt.show()
    
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
    distribution = johnson_SB
    
    path = path = "C: / Users / USUARIO1 / Desktop / Fitter / data / data_exponencial.txt"
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