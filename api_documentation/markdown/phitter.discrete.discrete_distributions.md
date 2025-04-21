* [phitter](modules.html)
* [phitter package](phitter.html)
* [phitter.discrete package](phitter.discrete.html)
* phitter.discrete.discrete\_distributions package
* [View page source](_sources/phitter.discrete.discrete_distributions.rst.txt)

---

# phitter.discrete.discrete\_distributions package[](#phitter-discrete-discrete-distributions-package "Link to this heading")

## Submodules[](#submodules "Link to this heading")

## phitter.discrete.discrete\_distributions.bernoulli module[](#module-phitter.discrete.discrete_distributions.bernoulli "Link to this heading")

*class* phitter.discrete.discrete\_distributions.bernoulli.Bernoulli(*parameters=None*, *discrete\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.discrete.discrete_distributions.bernoulli.Bernoulli "Link to this definition")
:   Bases: `object`

## Bernoulli distribution
## - Parameters Bernoulli Distribution: {“p”: \*}
## - <https://phitter.io/distributions/discrete/bernoulli>

## cdf(*x*)[](#phitter.discrete.discrete_distributions.bernoulli.Bernoulli.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.discrete.discrete_distributions.bernoulli.Bernoulli.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*discrete\_measures*)[](#phitter.discrete.discrete_distributions.bernoulli.Bernoulli.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample discrete\_measures.
- The parameters are calculated by formula.

#### Parameters
**discrete\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“p”: \*}

## *property* kurtosis*: float*[](#phitter.discrete.discrete_distributions.bernoulli.Bernoulli.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.discrete.discrete_distributions.bernoulli.Bernoulli.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.discrete.discrete_distributions.bernoulli.Bernoulli.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.discrete.discrete_distributions.bernoulli.Bernoulli.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.discrete.discrete_distributions.bernoulli.Bernoulli.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.discrete.discrete_distributions.bernoulli.Bernoulli.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.discrete.discrete_distributions.bernoulli.Bernoulli.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.discrete.discrete_distributions.bernoulli.Bernoulli.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.discrete.discrete_distributions.bernoulli.Bernoulli.parameters_example "Link to this definition")

## pmf(*x*)[](#phitter.discrete.discrete_distributions.bernoulli.Bernoulli.pmf "Link to this definition")
Probability mass function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.discrete.discrete_distributions.bernoulli.Bernoulli.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.discrete.discrete_distributions.bernoulli.Bernoulli.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.discrete.discrete_distributions.bernoulli.Bernoulli.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.discrete.discrete_distributions.bernoulli.Bernoulli.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.discrete.discrete_distributions.bernoulli.Bernoulli.variance "Link to this definition")
Parametric variance

## phitter.discrete.discrete\_distributions.binomial module[](#module-phitter.discrete.discrete_distributions.binomial "Link to this heading")

*class* phitter.discrete.discrete\_distributions.binomial.Binomial(*parameters=None*, *discrete\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.discrete.discrete_distributions.binomial.Binomial "Link to this definition")
:   Bases: `object`

## Binomial distribution
## - Parameters Binomial Distribution: {“n”: \*, “p”: \*}
## - <https://phitter.io/distributions/discrete/binomial>

## cdf(*x*)[](#phitter.discrete.discrete_distributions.binomial.Binomial.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.discrete.discrete_distributions.binomial.Binomial.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*discrete\_measures*)[](#phitter.discrete.discrete_distributions.binomial.Binomial.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample discrete\_measures.
- The parameters are calculated by formula.

#### Parameters
**discrete\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“n”: \*, “p”: \*}

## *property* kurtosis*: float*[](#phitter.discrete.discrete_distributions.binomial.Binomial.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.discrete.discrete_distributions.binomial.Binomial.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.discrete.discrete_distributions.binomial.Binomial.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.discrete.discrete_distributions.binomial.Binomial.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.discrete.discrete_distributions.binomial.Binomial.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.discrete.discrete_distributions.binomial.Binomial.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.discrete.discrete_distributions.binomial.Binomial.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.discrete.discrete_distributions.binomial.Binomial.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.discrete.discrete_distributions.binomial.Binomial.parameters_example "Link to this definition")

## pmf(*x*)[](#phitter.discrete.discrete_distributions.binomial.Binomial.pmf "Link to this definition")
Probability mass function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.discrete.discrete_distributions.binomial.Binomial.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.discrete.discrete_distributions.binomial.Binomial.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.discrete.discrete_distributions.binomial.Binomial.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.discrete.discrete_distributions.binomial.Binomial.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.discrete.discrete_distributions.binomial.Binomial.variance "Link to this definition")
Parametric variance

## phitter.discrete.discrete\_distributions.geometric module[](#module-phitter.discrete.discrete_distributions.geometric "Link to this heading")

*class* phitter.discrete.discrete\_distributions.geometric.Geometric(*parameters=None*, *discrete\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.discrete.discrete_distributions.geometric.Geometric "Link to this definition")
:   Bases: `object`

## Geometric distribution
## - Parameters Geometric Distribution: {“p”: \*}
## - <https://phitter.io/distributions/discrete/geometric>

## cdf(*x*)[](#phitter.discrete.discrete_distributions.geometric.Geometric.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.discrete.discrete_distributions.geometric.Geometric.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*discrete\_measures*)[](#phitter.discrete.discrete_distributions.geometric.Geometric.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample discrete\_measures.
- The parameters are calculated by formula.

#### Parameters
**discrete\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“p”: \*}

## *property* kurtosis*: float*[](#phitter.discrete.discrete_distributions.geometric.Geometric.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.discrete.discrete_distributions.geometric.Geometric.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.discrete.discrete_distributions.geometric.Geometric.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.discrete.discrete_distributions.geometric.Geometric.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.discrete.discrete_distributions.geometric.Geometric.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.discrete.discrete_distributions.geometric.Geometric.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.discrete.discrete_distributions.geometric.Geometric.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.discrete.discrete_distributions.geometric.Geometric.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.discrete.discrete_distributions.geometric.Geometric.parameters_example "Link to this definition")

## pmf(*x*)[](#phitter.discrete.discrete_distributions.geometric.Geometric.pmf "Link to this definition")
Probability mass function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.discrete.discrete_distributions.geometric.Geometric.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.discrete.discrete_distributions.geometric.Geometric.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.discrete.discrete_distributions.geometric.Geometric.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.discrete.discrete_distributions.geometric.Geometric.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.discrete.discrete_distributions.geometric.Geometric.variance "Link to this definition")
Parametric variance

## phitter.discrete.discrete\_distributions.hypergeometric module[](#module-phitter.discrete.discrete_distributions.hypergeometric "Link to this heading")

*class* phitter.discrete.discrete\_distributions.hypergeometric.Hypergeometric(*parameters=None*, *discrete\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.discrete.discrete_distributions.hypergeometric.Hypergeometric "Link to this definition")
:   Bases: `object`

## Hypergeometric\_distribution
## - Parameters Hypergeometric Distribution: {“N”: \*, “K”: \*, “n”: \*}
## - <https://phitter.io/distributions/discrete/hypergeometric>

## cdf(*x*)[](#phitter.discrete.discrete_distributions.hypergeometric.Hypergeometric.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.discrete.discrete_distributions.hypergeometric.Hypergeometric.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*discrete\_measures*)[](#phitter.discrete.discrete_distributions.hypergeometric.Hypergeometric.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample discrete\_measures.
- The parameters are calculated by formula.

#### Parameters
**discrete\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“N”: \*, “K”: \*, “n”: \*}

## *property* kurtosis*: float*[](#phitter.discrete.discrete_distributions.hypergeometric.Hypergeometric.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.discrete.discrete_distributions.hypergeometric.Hypergeometric.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.discrete.discrete_distributions.hypergeometric.Hypergeometric.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.discrete.discrete_distributions.hypergeometric.Hypergeometric.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.discrete.discrete_distributions.hypergeometric.Hypergeometric.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.discrete.discrete_distributions.hypergeometric.Hypergeometric.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.discrete.discrete_distributions.hypergeometric.Hypergeometric.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.discrete.discrete_distributions.hypergeometric.Hypergeometric.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.discrete.discrete_distributions.hypergeometric.Hypergeometric.parameters_example "Link to this definition")

## pmf(*x*)[](#phitter.discrete.discrete_distributions.hypergeometric.Hypergeometric.pmf "Link to this definition")
Probability mass function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.discrete.discrete_distributions.hypergeometric.Hypergeometric.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.discrete.discrete_distributions.hypergeometric.Hypergeometric.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.discrete.discrete_distributions.hypergeometric.Hypergeometric.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.discrete.discrete_distributions.hypergeometric.Hypergeometric.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.discrete.discrete_distributions.hypergeometric.Hypergeometric.variance "Link to this definition")
Parametric variance

## phitter.discrete.discrete\_distributions.logarithmic module[](#module-phitter.discrete.discrete_distributions.logarithmic "Link to this heading")

*class* phitter.discrete.discrete\_distributions.logarithmic.Logarithmic(*parameters=None*, *discrete\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.discrete.discrete_distributions.logarithmic.Logarithmic "Link to this definition")
:   Bases: `object`

## Logarithmic distribution
## - Parameters Logarithmic Distribution: {“p”: \*}
## - <https://phitter.io/distributions/discrete/logarithmic>

## cdf(*x*)[](#phitter.discrete.discrete_distributions.logarithmic.Logarithmic.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.discrete.discrete_distributions.logarithmic.Logarithmic.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*discrete\_measures*)[](#phitter.discrete.discrete_distributions.logarithmic.Logarithmic.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample discrete\_measures.
- The parameters are calculated by formula.

#### Parameters
**discrete\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“p”: \*}

## *property* kurtosis*: float*[](#phitter.discrete.discrete_distributions.logarithmic.Logarithmic.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.discrete.discrete_distributions.logarithmic.Logarithmic.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.discrete.discrete_distributions.logarithmic.Logarithmic.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.discrete.discrete_distributions.logarithmic.Logarithmic.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.discrete.discrete_distributions.logarithmic.Logarithmic.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.discrete.discrete_distributions.logarithmic.Logarithmic.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.discrete.discrete_distributions.logarithmic.Logarithmic.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.discrete.discrete_distributions.logarithmic.Logarithmic.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.discrete.discrete_distributions.logarithmic.Logarithmic.parameters_example "Link to this definition")

## pmf(*x*)[](#phitter.discrete.discrete_distributions.logarithmic.Logarithmic.pmf "Link to this definition")
Probability mass function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.discrete.discrete_distributions.logarithmic.Logarithmic.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.discrete.discrete_distributions.logarithmic.Logarithmic.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.discrete.discrete_distributions.logarithmic.Logarithmic.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.discrete.discrete_distributions.logarithmic.Logarithmic.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.discrete.discrete_distributions.logarithmic.Logarithmic.variance "Link to this definition")
Parametric variance

## phitter.discrete.discrete\_distributions.negative\_binomial module[](#module-phitter.discrete.discrete_distributions.negative_binomial "Link to this heading")

*class* phitter.discrete.discrete\_distributions.negative\_binomial.NegativeBinomial(*parameters=None*, *discrete\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.discrete.discrete_distributions.negative_binomial.NegativeBinomial "Link to this definition")
:   Bases: `object`

## Negative binomial distribution
## - Parameters NegativeBinomial Distribution: {“r”: \*, “p”: \*}
## - <https://phitter.io/distributions/discrete/negative_binomial>

## cdf(*x*)[](#phitter.discrete.discrete_distributions.negative_binomial.NegativeBinomial.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.discrete.discrete_distributions.negative_binomial.NegativeBinomial.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*discrete\_measures*)[](#phitter.discrete.discrete_distributions.negative_binomial.NegativeBinomial.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample discrete\_measures.
- The parameters are calculated by formula.

#### Parameters
**discrete\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“r”: \*, “p”: \*}

## *property* kurtosis*: float*[](#phitter.discrete.discrete_distributions.negative_binomial.NegativeBinomial.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.discrete.discrete_distributions.negative_binomial.NegativeBinomial.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.discrete.discrete_distributions.negative_binomial.NegativeBinomial.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.discrete.discrete_distributions.negative_binomial.NegativeBinomial.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.discrete.discrete_distributions.negative_binomial.NegativeBinomial.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.discrete.discrete_distributions.negative_binomial.NegativeBinomial.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.discrete.discrete_distributions.negative_binomial.NegativeBinomial.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.discrete.discrete_distributions.negative_binomial.NegativeBinomial.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.discrete.discrete_distributions.negative_binomial.NegativeBinomial.parameters_example "Link to this definition")

## pmf(*x*)[](#phitter.discrete.discrete_distributions.negative_binomial.NegativeBinomial.pmf "Link to this definition")
Probability mass function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.discrete.discrete_distributions.negative_binomial.NegativeBinomial.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.discrete.discrete_distributions.negative_binomial.NegativeBinomial.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.discrete.discrete_distributions.negative_binomial.NegativeBinomial.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.discrete.discrete_distributions.negative_binomial.NegativeBinomial.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.discrete.discrete_distributions.negative_binomial.NegativeBinomial.variance "Link to this definition")
Parametric variance

## phitter.discrete.discrete\_distributions.poisson module[](#module-phitter.discrete.discrete_distributions.poisson "Link to this heading")

*class* phitter.discrete.discrete\_distributions.poisson.Poisson(*parameters=None*, *discrete\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.discrete.discrete_distributions.poisson.Poisson "Link to this definition")
:   Bases: `object`

## Poisson distribution
## - Parameters Poisson Distribution: {“lambda”: \*}
## - <https://phitter.io/distributions/discrete/poisson>

## cdf(*x*)[](#phitter.discrete.discrete_distributions.poisson.Poisson.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.discrete.discrete_distributions.poisson.Poisson.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*discrete\_measures*)[](#phitter.discrete.discrete_distributions.poisson.Poisson.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample discrete\_measures.
- The parameters are calculated by formula.

#### Parameters
**discrete\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“lambda”: \*}

## *property* kurtosis*: float*[](#phitter.discrete.discrete_distributions.poisson.Poisson.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.discrete.discrete_distributions.poisson.Poisson.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.discrete.discrete_distributions.poisson.Poisson.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.discrete.discrete_distributions.poisson.Poisson.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.discrete.discrete_distributions.poisson.Poisson.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.discrete.discrete_distributions.poisson.Poisson.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.discrete.discrete_distributions.poisson.Poisson.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.discrete.discrete_distributions.poisson.Poisson.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.discrete.discrete_distributions.poisson.Poisson.parameters_example "Link to this definition")

## pmf(*x*)[](#phitter.discrete.discrete_distributions.poisson.Poisson.pmf "Link to this definition")
Probability mass function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.discrete.discrete_distributions.poisson.Poisson.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.discrete.discrete_distributions.poisson.Poisson.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.discrete.discrete_distributions.poisson.Poisson.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.discrete.discrete_distributions.poisson.Poisson.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.discrete.discrete_distributions.poisson.Poisson.variance "Link to this definition")
Parametric variance

## phitter.discrete.discrete\_distributions.uniform module[](#module-phitter.discrete.discrete_distributions.uniform "Link to this heading")

*class* phitter.discrete.discrete\_distributions.uniform.Uniform(*parameters=None*, *discrete\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.discrete.discrete_distributions.uniform.Uniform "Link to this definition")
:   Bases: `object`

## Uniform distribution
## - Parameters Uniform Distribution: {“a”: \*, “b”: \*}
## - <https://phitter.io/distributions/discrete/uniform>

## cdf(*x*)[](#phitter.discrete.discrete_distributions.uniform.Uniform.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.discrete.discrete_distributions.uniform.Uniform.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*discrete\_measures*)[](#phitter.discrete.discrete_distributions.uniform.Uniform.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample discrete\_measures.
- The parameters are calculated by formula.

#### Parameters
**discrete\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“a”: \*, “b”: \*}

## *property* kurtosis*: float*[](#phitter.discrete.discrete_distributions.uniform.Uniform.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.discrete.discrete_distributions.uniform.Uniform.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.discrete.discrete_distributions.uniform.Uniform.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.discrete.discrete_distributions.uniform.Uniform.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.discrete.discrete_distributions.uniform.Uniform.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.discrete.discrete_distributions.uniform.Uniform.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.discrete.discrete_distributions.uniform.Uniform.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.discrete.discrete_distributions.uniform.Uniform.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.discrete.discrete_distributions.uniform.Uniform.parameters_example "Link to this definition")

## pmf(*x*)[](#phitter.discrete.discrete_distributions.uniform.Uniform.pmf "Link to this definition")
Probability mass function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.discrete.discrete_distributions.uniform.Uniform.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.discrete.discrete_distributions.uniform.Uniform.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.discrete.discrete_distributions.uniform.Uniform.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.discrete.discrete_distributions.uniform.Uniform.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.discrete.discrete_distributions.uniform.Uniform.variance "Link to this definition")
Parametric variance

## Module contents[](#module-phitter.discrete.discrete_distributions "Link to this heading")

[Previous](phitter.discrete.html "phitter.discrete package")
[Next](phitter.discrete.discrete_measures.html "phitter.discrete.discrete_measures package")

---

© Copyright .

Built with [Sphinx](https://www.sphinx-doc.org/) using a
[theme](https://github.com/readthedocs/sphinx_rtd_theme)
provided by [Read the Docs](https://readthedocs.org).