import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy
import scipy.stats

mpl.style.use("ggplot")


def plot_histogram(data, distribution):
    plt.figure(figsize=(8, 4))
    plt.hist(data, density=True, ec="white", bins=20)
    plt.title("HISTOGRAM")
    plt.xlabel("Values")
    plt.ylabel("Frequencies")

    x_plot = numpy.linspace(min(data), max(data), 1000)
    y_plot = distribution.pdf(x_plot)
    plt.plot(x_plot, y_plot, label="PDF KDE")

    plt.legend(title="DISTRIBUTIONS", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.show()


def get_data(path: str) -> list[float | int]:
    sample_distribution_file = open(path, "r")
    data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
    return data


if __name__ == "__main__":
    ## Get Data
    path = "../../datasets_test/continuous/data_1000/data_beta.txt"
    data = get_data(path)

    ## KDE
    distribution = scipy.stats.gaussian_kde(data)

    ## Plot result
    plot_histogram(data, distribution)

    ## Prove PDF distribution
    print(distribution.pdf(50000))
    print(distribution.evaluate(50000))
