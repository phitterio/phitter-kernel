* [phitter](modules.html)
* [phitter package](phitter.html)
* [phitter.discrete package](phitter.discrete.html)
* phitter.discrete.discrete\_distributions package
* [View page source](_sources/phitter.discrete.discrete_distributions.rst.txt)

---

# phitter.discrete.discrete\_distributions package

## Submodules

## phitter.discrete.discrete\_distributions.bernoulli module

*class* phitter.discrete.discrete\_distributions.bernoulli.Bernoulli(*parameters=None*, *discrete\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Bernoulli distribution
## - Parameters Bernoulli Distribution: {“p”: \*}
## - <https://phitter.io/distributions/discrete/bernoulli>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*discrete\_measures*)
Calculate proper parameters of the distribution from sample discrete\_measures.
- The parameters are calculated by formula.

#### Parameters
**discrete\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“p”: \*}

## *property* kurtosis*: float*
Parametric kurtosis

## *property* mean*: float*
Parametric mean

## *property* median*: float*
Parametric median

## *property* mode*: float*
Parametric mode

## *property* name

## non\_central\_moments(*k*)
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*
Number of parameters of the distribution

## parameter\_restrictions()
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*

## pmf(*x*)
Probability mass function

#### Return type
`float` | `ndarray`

## ppf(*u*)
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*
Parametric skewness

## *property* standard\_deviation*: float*
Parametric standard deviation

## *property* variance*: float*
Parametric variance

## phitter.discrete.discrete\_distributions.binomial module

*class* phitter.discrete.discrete\_distributions.binomial.Binomial(*parameters=None*, *discrete\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Binomial distribution
## - Parameters Binomial Distribution: {“n”: \*, “p”: \*}
## - <https://phitter.io/distributions/discrete/binomial>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*discrete\_measures*)
Calculate proper parameters of the distribution from sample discrete\_measures.
- The parameters are calculated by formula.

#### Parameters
**discrete\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“n”: \*, “p”: \*}

## *property* kurtosis*: float*
Parametric kurtosis

## *property* mean*: float*
Parametric mean

## *property* median*: float*
Parametric median

## *property* mode*: float*
Parametric mode

## *property* name

## non\_central\_moments(*k*)
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*
Number of parameters of the distribution

## parameter\_restrictions()
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*

## pmf(*x*)
Probability mass function

#### Return type
`float` | `ndarray`

## ppf(*u*)
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*
Parametric skewness

## *property* standard\_deviation*: float*
Parametric standard deviation

## *property* variance*: float*
Parametric variance

## phitter.discrete.discrete\_distributions.geometric module

*class* phitter.discrete.discrete\_distributions.geometric.Geometric(*parameters=None*, *discrete\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Geometric distribution
## - Parameters Geometric Distribution: {“p”: \*}
## - <https://phitter.io/distributions/discrete/geometric>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*discrete\_measures*)
Calculate proper parameters of the distribution from sample discrete\_measures.
- The parameters are calculated by formula.

#### Parameters
**discrete\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“p”: \*}

## *property* kurtosis*: float*
Parametric kurtosis

## *property* mean*: float*
Parametric mean

## *property* median*: float*
Parametric median

## *property* mode*: float*
Parametric mode

## *property* name

## non\_central\_moments(*k*)
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*
Number of parameters of the distribution

## parameter\_restrictions()
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*

## pmf(*x*)
Probability mass function

#### Return type
`float` | `ndarray`

## ppf(*u*)
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*
Parametric skewness

## *property* standard\_deviation*: float*
Parametric standard deviation

## *property* variance*: float*
Parametric variance

## phitter.discrete.discrete\_distributions.hypergeometric module

*class* phitter.discrete.discrete\_distributions.hypergeometric.Hypergeometric(*parameters=None*, *discrete\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Hypergeometric\_distribution
## - Parameters Hypergeometric Distribution: {“N”: \*, “K”: \*, “n”: \*}
## - <https://phitter.io/distributions/discrete/hypergeometric>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*discrete\_measures*)
Calculate proper parameters of the distribution from sample discrete\_measures.
- The parameters are calculated by formula.

#### Parameters
**discrete\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“N”: \*, “K”: \*, “n”: \*}

## *property* kurtosis*: float*
Parametric kurtosis

## *property* mean*: float*
Parametric mean

## *property* median*: float*
Parametric median

## *property* mode*: float*
Parametric mode

## *property* name

## non\_central\_moments(*k*)
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*
Number of parameters of the distribution

## parameter\_restrictions()
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*

## pmf(*x*)
Probability mass function

#### Return type
`float` | `ndarray`

## ppf(*u*)
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*
Parametric skewness

## *property* standard\_deviation*: float*
Parametric standard deviation

## *property* variance*: float*
Parametric variance

## phitter.discrete.discrete\_distributions.logarithmic module

*class* phitter.discrete.discrete\_distributions.logarithmic.Logarithmic(*parameters=None*, *discrete\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Logarithmic distribution
## - Parameters Logarithmic Distribution: {“p”: \*}
## - <https://phitter.io/distributions/discrete/logarithmic>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*discrete\_measures*)
Calculate proper parameters of the distribution from sample discrete\_measures.
- The parameters are calculated by formula.

#### Parameters
**discrete\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“p”: \*}

## *property* kurtosis*: float*
Parametric kurtosis

## *property* mean*: float*
Parametric mean

## *property* median*: float*
Parametric median

## *property* mode*: float*
Parametric mode

## *property* name

## non\_central\_moments(*k*)
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*
Number of parameters of the distribution

## parameter\_restrictions()
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*

## pmf(*x*)
Probability mass function

#### Return type
`float` | `ndarray`

## ppf(*u*)
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*
Parametric skewness

## *property* standard\_deviation*: float*
Parametric standard deviation

## *property* variance*: float*
Parametric variance

## phitter.discrete.discrete\_distributions.negative\_binomial module

*class* phitter.discrete.discrete\_distributions.negative\_binomial.NegativeBinomial(*parameters=None*, *discrete\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Negative binomial distribution
## - Parameters NegativeBinomial Distribution: {“r”: \*, “p”: \*}
## - <https://phitter.io/distributions/discrete/negative_binomial>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*discrete\_measures*)
Calculate proper parameters of the distribution from sample discrete\_measures.
- The parameters are calculated by formula.

#### Parameters
**discrete\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“r”: \*, “p”: \*}

## *property* kurtosis*: float*
Parametric kurtosis

## *property* mean*: float*
Parametric mean

## *property* median*: float*
Parametric median

## *property* mode*: float*
Parametric mode

## *property* name

## non\_central\_moments(*k*)
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*
Number of parameters of the distribution

## parameter\_restrictions()
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*

## pmf(*x*)
Probability mass function

#### Return type
`float` | `ndarray`

## ppf(*u*)
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*
Parametric skewness

## *property* standard\_deviation*: float*
Parametric standard deviation

## *property* variance*: float*
Parametric variance

## phitter.discrete.discrete\_distributions.poisson module

*class* phitter.discrete.discrete\_distributions.poisson.Poisson(*parameters=None*, *discrete\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Poisson distribution
## - Parameters Poisson Distribution: {“lambda”: \*}
## - <https://phitter.io/distributions/discrete/poisson>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*discrete\_measures*)
Calculate proper parameters of the distribution from sample discrete\_measures.
- The parameters are calculated by formula.

#### Parameters
**discrete\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“lambda”: \*}

## *property* kurtosis*: float*
Parametric kurtosis

## *property* mean*: float*
Parametric mean

## *property* median*: float*
Parametric median

## *property* mode*: float*
Parametric mode

## *property* name

## non\_central\_moments(*k*)
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*
Number of parameters of the distribution

## parameter\_restrictions()
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*

## pmf(*x*)
Probability mass function

#### Return type
`float` | `ndarray`

## ppf(*u*)
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*
Parametric skewness

## *property* standard\_deviation*: float*
Parametric standard deviation

## *property* variance*: float*
Parametric variance

## phitter.discrete.discrete\_distributions.uniform module

*class* phitter.discrete.discrete\_distributions.uniform.Uniform(*parameters=None*, *discrete\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Uniform distribution
## - Parameters Uniform Distribution: {“a”: \*, “b”: \*}
## - <https://phitter.io/distributions/discrete/uniform>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*discrete\_measures*)
Calculate proper parameters of the distribution from sample discrete\_measures.
- The parameters are calculated by formula.

#### Parameters
**discrete\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“a”: \*, “b”: \*}

## *property* kurtosis*: float*
Parametric kurtosis

## *property* mean*: float*
Parametric mean

## *property* median*: float*
Parametric median

## *property* mode*: float*
Parametric mode

## *property* name

## non\_central\_moments(*k*)
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*
Number of parameters of the distribution

## parameter\_restrictions()
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*

## pmf(*x*)
Probability mass function

#### Return type
`float` | `ndarray`

## ppf(*u*)
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*
Parametric skewness

## *property* standard\_deviation*: float*
Parametric standard deviation

## *property* variance*: float*
Parametric variance

## Module contents

[Previous](phitter.discrete.html "phitter.discrete package")
[Next](phitter.discrete.discrete_measures.html "phitter.discrete.discrete_measures package")

---

© Copyright .

Built with [Sphinx](https://www.sphinx-doc.org/) using a
[theme](https://github.com/readthedocs/sphinx_rtd_theme)
provided by [Read the Docs](https://readthedocs.org).