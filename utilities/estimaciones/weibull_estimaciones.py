from scipy.optimize import fsolve
import numpy

def equations(initial_solution, p):
    alpha_, beta_ = initial_solution
    _mean, _variance = p
    
    ## Mean
    f1 = (beta_ / alpha_) * scipy.special.gamma(1 / alpha_) - _mean
    ## Variance
    f2 = (beta_ ** 2 / alpha_) * (2 * scipy.special.gamma(2 / alpha_) - (1 / alpha_) * scipy.special.gamma(1 / alpha_) ** 2) - _variance
   
    return (f1, f2)

sol = fsolve(equations, (1, 1), [9.35278469, 2.323387311])
print(sol)