from .bernoulli import BERNOULLI
from .binomial import BINOMIAL
from .geometric import GEOMETRIC
from .hypergeometric import HYPERGEOMETRIC
from .logarithmic import LOGARITHMIC
from .negative_binomial import NEGATIVE_BINOMIAL
from .poisson import POISSON
from .uniform import UNIFORM

DISCRETE_DISTRIBUTIONS = {
    "bernoulli": BERNOULLI,
    "binomial": BINOMIAL,
    "geometric": GEOMETRIC,
    "hypergeometric": HYPERGEOMETRIC,
    "logarithmic": LOGARITHMIC,
    "negative_binomial": NEGATIVE_BINOMIAL,
    "poisson": POISSON,
    "uniform": UNIFORM,
}
