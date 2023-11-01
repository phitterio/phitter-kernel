import scipy.stats
import random
file = open("../data/data_generalized_gamma.txt", "w")

a, c, mu, sigma = 15,  - 5, 100, 10
for _ in range(3000):
    p = random.uniform(0, 1)
    x = scipy.stats.gengamma.ppf(p, a, c, loc=mu, scale=sigma)
    file.write(str(x).replace(".", ",") + "\n")
p = random.random()
x = scipy.stats.gengamma.ppf(p, a, c, loc=mu, scale=sigma)
file.write(str(x).replace(".", ","))   
file.close()


def getData(direction):
    sample_distribution_file = open(direction, "r")
    data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
    return data

path = "../data/data_generalized_gamma.txt"
data = getData(path)

print(scipy.stats.gengamma.fit(data))