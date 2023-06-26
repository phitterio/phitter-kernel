import math
import scipy.stats

def doanes_formula(data):
    """
    DONAE'S FORMULA
    https://en.wikipedia.org/wiki/Histogram#Doane's_formula
    
    Parameters
    ==========
    data : iterable 
        data set
    Returns
     -  -  -  -  -  -  - 
    num_bins : int
        Cumulative distribution function evaluated at x
    """
    N = len(data)
    skewness = scipy.stats.skew(data)
    sigma_g1 = math.sqrt((6 * (N - 2)) / ((N + 1) * (N + 3)))
    num_bins = 1 + math.log(N,2) + math.log(1 + abs(skewness) / sigma_g1,2)
    num_bins = round(num_bins)
    return num_bins

if __name__ == "__main__":
    ## Import function to get measurements
    def get_data(direction):
        sample_distribution_file = open(direction, "r")
        data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
        return data
    
    ## Distribution class
    path = "../data/data_normal.txt"
    data = get_data(path) 
    
    print(doanes_formula(data))