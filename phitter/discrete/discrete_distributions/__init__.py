from .bernoulli import Bernoulli
from .binomial import Binomial
from .geometric import Geometric
from .hypergeometric import Hypergeometric
from .logarithmic import Logarithmic
from .negative_binomial import NegativeBinomial
from .poisson import Poisson
from .uniform import Uniform

DISCRETE_DISTRIBUTIONS = {
    "bernoulli": Bernoulli,
    "binomial": Binomial,
    "geometric": Geometric,
    "hypergeometric": Hypergeometric,
    "logarithmic": Logarithmic,
    "negative_binomial": NegativeBinomial,
    "poisson": Poisson,
    "uniform": Uniform,
}
