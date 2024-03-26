import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy
import pandas
import scipy.stats
import statsmodels.api as sm

mpl.style.use("ggplot")


def plot_histogram(data, results, n):
    ## n first distribution of the ranking
    N_DISTRIBUTIONS = {k: results[k] for k in list(results)[:n]}

    ## Histogram of data
    plt.figure(figsize=(8, 4))
    plt.hist(data, density=True, ec="white", color=(192 / 235, 17 / 235, 17 / 235))
    plt.title("HISTOGRAM")
    plt.xlabel("Values")
    plt.ylabel("Frequencies")

    ## Plot n distributions
    for distribution, result in N_DISTRIBUTIONS.items():
        sse = result[0]
        arg = result[1]
        loc = result[2]
        scale = result[3]
        x_plot = numpy.linspace(min(data), max(data), 1000)
        y_plot = distribution.pdf(x_plot, loc=loc, scale=scale, *arg)
        plt.plot(x_plot, y_plot, label=str(distribution)[32:-34] + ": " + str(sse)[0:6], color=(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))

    plt.legend(title="DISTRIBUTIONS", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.show()


def fit_data(data):
    ## scipy.stats.frechet_r,scipy.stats.frechet_l: are disbled in current SciPy version
    ## scipy.stats.levy_stable: a lot of time of estimation parameters
    ALL_DISTRIBUTIONS = [
        scipy.stats.alpha,
        scipy.stats.anglit,
        scipy.stats.arcsine,
        scipy.stats.beta,
        scipy.stats.betaprime,
        scipy.stats.bradford,
        scipy.stats.burr,
        scipy.stats.cauchy,
        scipy.stats.chi,
        scipy.stats.chi2,
        scipy.stats.cosine,
        scipy.stats.dgamma,
        scipy.stats.dweibull,
        scipy.stats.erlang,
        scipy.stats.expon,
        scipy.stats.exponnorm,
        scipy.stats.exponweib,
        scipy.stats.exponpow,
        scipy.stats.f,
        scipy.stats.fatiguelife,
        scipy.stats.fisk,
        scipy.stats.foldcauchy,
        scipy.stats.foldnorm,
        scipy.stats.genlogistic,
        scipy.stats.genpareto,
        scipy.stats.gennorm,
        scipy.stats.genexpon,
        scipy.stats.genextreme,
        scipy.stats.gausshyper,
        scipy.stats.gamma,
        scipy.stats.gengamma,
        scipy.stats.genhalflogistic,
        scipy.stats.gilbrat,
        scipy.stats.gompertz,
        scipy.stats.gumbel_r,
        scipy.stats.gumbel_l,
        scipy.stats.halfcauchy,
        scipy.stats.halflogistic,
        scipy.stats.halfnorm,
        scipy.stats.halfgennorm,
        scipy.stats.hypsecant,
        scipy.stats.invgamma,
        scipy.stats.invgauss,
        scipy.stats.invweibull,
        scipy.stats.johnsonsb,
        scipy.stats.johnsonsu,
        scipy.stats.ksone,
        scipy.stats.kstwobign,
        scipy.stats.laplace,
        scipy.stats.levy,
        scipy.stats.levy_l,
        scipy.stats.logistic,
        scipy.stats.loggamma,
        scipy.stats.loglaplace,
        scipy.stats.lognorm,
        scipy.stats.lomax,
        scipy.stats.maxwell,
        scipy.stats.mielke,
        scipy.stats.nakagami,
        scipy.stats.ncx2,
        scipy.stats.ncf,
        scipy.stats.nct,
        scipy.stats.norm,
        scipy.stats.pareto,
        scipy.stats.pearson3,
        scipy.stats.powerlaw,
        scipy.stats.powerlognorm,
        scipy.stats.powernorm,
        scipy.stats.rdist,
        scipy.stats.reciprocal,
        scipy.stats.rayleigh,
        scipy.stats.rice,
        scipy.stats.recipinvgauss,
        scipy.stats.semicircular,
        scipy.stats.t,
        scipy.stats.triang,
        scipy.stats.truncexpon,
        scipy.stats.truncnorm,
        scipy.stats.tukeylambda,
        scipy.stats.uniform,
        scipy.stats.vonmises,
        scipy.stats.vonmises_line,
        scipy.stats.wald,
        scipy.stats.weibull_min,
        scipy.stats.weibull_max,
        scipy.stats.wrapcauchy,
    ]

    # MY_DISTRIBUTIONS = [scipy.stats.beta, scipy.stats.expon, scipy.stats.norm, scipy.stats.uniform, scipy.stats.johnsonsb, scipy.stats.gennorm, scipy.stats.gausshyper, scipy.stats.gengamma]

    ## Calculae Histogram
    frequencies, bin_edges = numpy.histogram(data, density=True)
    central_values = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]

    results = {}
    for distribution in ALL_DISTRIBUTIONS:
        ## Get parameters of distribution
        params = distribution.fit(data)

        ## Separate parts of parameters
        args = params[:-2]
        loc = params[-2]
        scale = params[-1]

        ## Calculate fitted PDF and error with fit in distribution
        pdf_values = distribution.pdf(numpy.array(central_values), loc=loc, scale=scale, *args)

        ## Calculate SSE (sum of squared estimate of errors)
        sse = numpy.sum(numpy.power(frequencies - pdf_values, 2.0))

        ## Build results and sort by sse
        results[distribution.name] = {
            "distribution": distribution,
            "parameters": {"loc": loc, "scale": scale, "args": args},
            "sse": sse,
        }

    sorted_results = {dist_name: result for dist_name, result in sorted(results.items(), key=lambda x: x[1]["sse"])}
    return sorted_results


if __name__ == "__main__":
    ## Import data
    data = pandas.Series(sm.datasets.elnino.load_pandas().data.set_index("YEAR").values.ravel())

    def get_data(path: str) -> list[float | int]:
        sample_distribution_file = open(path, "r")
        data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
        sample_distribution_file.close()
        return data

    path = "./datasets_test/data_tescipy.stats.txt"
    data = get_data(path)

    results = fit_data(data)
    plot_histogram(data, results, 5)
