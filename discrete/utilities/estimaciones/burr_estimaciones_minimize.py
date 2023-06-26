from scipy.optimize import minimize
import math
from scipy.special import beta

def modificated_beta(x, y, infinite):
    if beta(x,y) == float('inf'):
        return infinite
    elif beta(x,y) == float(' - inf'):
        return  - infinite
    else:
        return beta(x,y)


def objective(initial_solution, p):
    A, B, C = initial_solution
    return A + B + C

def constraint_mean(initial_solution, data_mean):
    A, B, C = initial_solution
    miu = lambda r: (A ** r) * C * modificated_beta((B * C - r) / B, (B + r) / B, 1e7)
    return miu(1) - data_mean

def constraint_variance(initial_solution, data_variance):
    A, B, C = initial_solution
    miu = lambda r: (A ** r) * C * modificated_beta((B * C - r) / B, (B + r) / B, 1e7)
    return  - (miu(1) ** 2) + miu(2) - data_variance

def constraint_median(initial_solution, data_median):
    A, B, C = initial_solution
    return A * ((2 ** (1 / C)) - 1) ** (1 / B) - data_median

#############################################################################
## Import function to get measurements
import sys
sys.path.append("C: / Users / USUARIO / Desktop / Fitter / utilities")
from measurements.measurements import MEASUREMENTS

## Import function to get measurements
def get_data(direction):
    sample_distribution_file = open(direction, "r")
    data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
    return data

## Distribution class
path = "C: / Users / USUARIO / Desktop / Fitter / data / data_burr.txt"
data = get_data(path) 
measurements = MEASUREMENTS(data)
#############################################################################


bnds = [(1,1000),(1,1000),(1,1000)]
con1 = {"type":"eq", "fun": constraint_mean, "args":(measurements.mean,)}
con2 = {"type":"eq", "fun": constraint_variance, "args":(measurements.variance,)}
con3 = {"type":"eq", "fun": constraint_median, "args":(measurements.median,)}

solution = minimize(objective, [1,1,1], args = ("parameter_objective"), method="SLSQP", bounds = bnds, constraints = [con1, con2, con3])
parameters = {"A": solution.x[0], "B": solution.x[1], "C": solution.x[2]}
print(parameters)






def solution_error(parameters, measurements):
    A, B, C = parameters["A"], parameters["B"], parameters["C"]

    miu = lambda r: (A ** r) * C * beta((B * C - r) / B, (B + r) / B)
    
    # Parametric expected expressions
    parametric_mean = miu(1)
    parametric_variance = -(miu(1) ** 2) + miu(2)
    parametric_kurtosis = -3 * miu(1) ** 4 + 6 * miu(1) ** 2 * miu(2) -4 * miu(1) * miu(3) + miu(4)
    parametric_skewness = 2 * miu(1) ** 3 - 3 * miu(1) * miu(2) + miu(3)
    parametric_median = A * ((2 ** (1 / C)) - 1) ** (1 / B)
    print(parametric_mean, parametric_variance, parametric_median)
    print(measurements.mean, measurements.variance, measurements.median)
    
    error1 = parametric_mean - measurements.mean
    error2 = parametric_variance - measurements.variance
    error3 = parametric_skewness - measurements.skewness
    error4 = parametric_kurtosis - measurements.kurtosis
    error5 = parametric_median - measurements.median
    
    total_error = abs(error1) + abs(error2) + abs(error3) + abs(error4) + abs(error5)
    return total_error
print(solution_error(parameters, measurements))

print(" -  -  - ")
import scipy.stats
scipy_params = scipy.stats.burr12.fit(measurements.data)
parameters = {"A": scipy_params[3], "B": scipy_params[0], "C": scipy_params[1]}
print(solution_error(parameters, measurements))
# min_data = measurements.min
# max_data = measurements.max
# orders = [1e5, 1e6, 1e7, 1e8]
# contextual_orders = [o * min_data for o in orders] + [o * max_data for o in orders] 
  
# min_error = float("inf")
# for inf in contextual_orders:
#     bnds = [(1, max_data * 100),(1, max_data * 100),(1, max_data * 100)]
#     con1 = {"type":"eq", "fun": constraint_mean, "args": (inf,)}
#     con2 = {"type":"eq", "fun": constraint_variance, "args": (inf,)}
#     con3 = {"type":"eq", "fun": constraint_skewness, "args": (inf,)}
    
#     solution = minimize(objective, [1, 1, 1], method="SLSQP", bounds = bnds, constraints = [con1, con2, con3])
#     partial_parameters = {"A": solution.x[0], "B": solution.x[1], "C": solution.x[2]}
    
#     if solution_error(partial_parameters, measurements) < min_error:
#         min_error = solution_error(partial_parameters, measurements)
#         parameters = partial_parameters        


