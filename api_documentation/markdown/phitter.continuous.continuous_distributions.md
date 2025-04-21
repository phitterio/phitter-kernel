* [phitter](modules.html)
* [phitter package](phitter.html)
* [phitter.continuous package](phitter.continuous.html)
* phitter.continuous.continuous\_distributions package
* [View page source](_sources/phitter.continuous.continuous_distributions.rst.txt)

---

# phitter.continuous.continuous\_distributions package[](#phitter-continuous-continuous-distributions-package "Link to this heading")

## Submodules[](#submodules "Link to this heading")

## phitter.continuous.continuous\_distributions.alpha module[](#module-phitter.continuous.continuous_distributions.alpha "Link to this heading")

*class* phitter.continuous.continuous\_distributions.alpha.Alpha(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.alpha.Alpha "Link to this definition")
:   Bases: `object`

## Alpha distribution
## - Parameters Alpha Distribution: {“alpha”: \*, “loc”: \*, “scale”: \*}
## - <https://phitter.io/distributions/continuous/alpha>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.alpha.Alpha.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.alpha.Alpha.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.alpha.Alpha.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by solving the equations of the measures expected
- for this distribution.The number of equations to consider is equal to the number
- of parameters.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“alpha”: \*, “loc”: \*, “scale”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.alpha.Alpha.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.alpha.Alpha.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.alpha.Alpha.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.alpha.Alpha.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.alpha.Alpha.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.alpha.Alpha.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.alpha.Alpha.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.alpha.Alpha.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.alpha.Alpha.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.alpha.Alpha.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.alpha.Alpha.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.alpha.Alpha.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.alpha.Alpha.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.alpha.Alpha.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.alpha.Alpha.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.arcsine module[](#module-phitter.continuous.continuous_distributions.arcsine "Link to this heading")

*class* phitter.continuous.continuous\_distributions.arcsine.Arcsine(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.arcsine.Arcsine "Link to this definition")
:   Bases: `object`

## Arcsine distribution
## - Parameters Arcsine Distribution: {“a”: \*, “b”: \*}
## - <https://phitter.io/distributions/continuous/arcsine>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.arcsine.Arcsine.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.arcsine.Arcsine.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.arcsine.Arcsine.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“a”: \*, “b”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.arcsine.Arcsine.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.arcsine.Arcsine.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.arcsine.Arcsine.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.arcsine.Arcsine.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.arcsine.Arcsine.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.arcsine.Arcsine.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.arcsine.Arcsine.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.arcsine.Arcsine.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.arcsine.Arcsine.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.arcsine.Arcsine.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.arcsine.Arcsine.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.arcsine.Arcsine.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.arcsine.Arcsine.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.arcsine.Arcsine.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.arcsine.Arcsine.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.argus module[](#module-phitter.continuous.continuous_distributions.argus "Link to this heading")

*class* phitter.continuous.continuous\_distributions.argus.Argus(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.argus.Argus "Link to this definition")
:   Bases: `object`

## Argus distribution
## - Parameters Argus Distribution: {“chi”: \*, “loc”: \*, “scale”: \*}
## - <https://phitter.io/distributions/continuous/argus>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.argus.Argus.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.argus.Argus.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.argus.Argus.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by solving the equations of the measures expected
- for this distribution.The number of equations to consider is equal to the number
- of parameters.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“chi”: \*, “loc”: \*, “scale”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.argus.Argus.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.argus.Argus.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.argus.Argus.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.argus.Argus.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.argus.Argus.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.argus.Argus.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.argus.Argus.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.argus.Argus.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.argus.Argus.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.argus.Argus.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.argus.Argus.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.argus.Argus.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.argus.Argus.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.argus.Argus.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.argus.Argus.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.beta module[](#module-phitter.continuous.continuous_distributions.beta "Link to this heading")

*class* phitter.continuous.continuous\_distributions.beta.Beta(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.beta.Beta "Link to this definition")
:   Bases: `object`

## Beta distribution
## - Parameters Beta Distribution: {“alpha”: \*, “beta”: \*, “A”: \*, “B”: \*}
## - <https://phitter.io/distributions/continuous/beta>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.beta.Beta.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.beta.Beta.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.beta.Beta.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by solving the equations of the measures expected
- for this distribution.The number of equations to consider is equal to the number
- of parameters.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“alpha”: \*, “beta”: \*, “A”: \*, “B”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.beta.Beta.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.beta.Beta.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.beta.Beta.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.beta.Beta.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.beta.Beta.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.beta.Beta.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.beta.Beta.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.beta.Beta.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.beta.Beta.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.beta.Beta.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.beta.Beta.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.beta.Beta.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.beta.Beta.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.beta.Beta.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.beta.Beta.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.beta\_prime module[](#module-phitter.continuous.continuous_distributions.beta_prime "Link to this heading")

*class* phitter.continuous.continuous\_distributions.beta\_prime.BetaPrime(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.beta_prime.BetaPrime "Link to this definition")
:   Bases: `object`

## Beta Prime Distribution
## - Parameters BetaPrime Distribution: {“alpha”: \*, “beta”: \*}
## - <https://phitter.io/distributions/continuous/beta_prime>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.beta_prime.BetaPrime.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.beta_prime.BetaPrime.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.beta_prime.BetaPrime.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by solving the equations of the measures expected
- for this distribution.The number of equations to consider is equal to the number
- of parameters.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“alpha”: \*, “beta”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.beta_prime.BetaPrime.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.beta_prime.BetaPrime.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.beta_prime.BetaPrime.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.beta_prime.BetaPrime.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.beta_prime.BetaPrime.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.beta_prime.BetaPrime.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.beta_prime.BetaPrime.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.beta_prime.BetaPrime.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.beta_prime.BetaPrime.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.beta_prime.BetaPrime.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.beta_prime.BetaPrime.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.beta_prime.BetaPrime.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.beta_prime.BetaPrime.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.beta_prime.BetaPrime.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.beta_prime.BetaPrime.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.beta\_prime\_4p module[](#module-phitter.continuous.continuous_distributions.beta_prime_4p "Link to this heading")

*class* phitter.continuous.continuous\_distributions.beta\_prime\_4p.BetaPrime4P(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.beta_prime_4p.BetaPrime4P "Link to this definition")
:   Bases: `object`

## Beta Prime 4P Distribution
## - Parameters BetaPrime4P Distribution: {“alpha”: \*, “beta”: \*, “loc”: \*, “scale”: \*}
## - <https://phitter.io/distributions/continuous/beta_prime_4p>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.beta_prime_4p.BetaPrime4P.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.beta_prime_4p.BetaPrime4P.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.beta_prime_4p.BetaPrime4P.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by solving the equations of the measures expected
- for this distribution.The number of equations to consider is equal to the number
- of parameters.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“alpha”: \*, “beta”: \*, “loc”: \*, “scale”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.beta_prime_4p.BetaPrime4P.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.beta_prime_4p.BetaPrime4P.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.beta_prime_4p.BetaPrime4P.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.beta_prime_4p.BetaPrime4P.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.beta_prime_4p.BetaPrime4P.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.beta_prime_4p.BetaPrime4P.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.beta_prime_4p.BetaPrime4P.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.beta_prime_4p.BetaPrime4P.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.beta_prime_4p.BetaPrime4P.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.beta_prime_4p.BetaPrime4P.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.beta_prime_4p.BetaPrime4P.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.beta_prime_4p.BetaPrime4P.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.beta_prime_4p.BetaPrime4P.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.beta_prime_4p.BetaPrime4P.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.beta_prime_4p.BetaPrime4P.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.bradford module[](#module-phitter.continuous.continuous_distributions.bradford "Link to this heading")

*class* phitter.continuous.continuous\_distributions.bradford.Bradford(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.bradford.Bradford "Link to this definition")
:   Bases: `object`

## Bradford distribution
## - Parameters Bradford Distribution: {“c”: \*, “min”: \*, “max”: \*}
## - <https://phitter.io/distributions/continuous/bradford>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.bradford.Bradford.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.bradford.Bradford.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.bradford.Bradford.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“c”: \*, “min”: \*, “max”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.bradford.Bradford.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.bradford.Bradford.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.bradford.Bradford.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.bradford.Bradford.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.bradford.Bradford.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.bradford.Bradford.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.bradford.Bradford.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.bradford.Bradford.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.bradford.Bradford.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.bradford.Bradford.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.bradford.Bradford.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.bradford.Bradford.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.bradford.Bradford.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.bradford.Bradford.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.bradford.Bradford.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.burr module[](#module-phitter.continuous.continuous_distributions.burr "Link to this heading")

*class* phitter.continuous.continuous\_distributions.burr.Burr(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.burr.Burr "Link to this definition")
:   Bases: `object`

## Burr distribution
## - Parameters Burr Distribution: {“A”: \*, “B”: \*, “C”: \*}
## - <https://phitter.io/distributions/continuous/burr>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.burr.Burr.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.burr.Burr.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.burr.Burr.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“A”: \*, “B”: \*, “C”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.burr.Burr.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.burr.Burr.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.burr.Burr.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.burr.Burr.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.burr.Burr.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.burr.Burr.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.burr.Burr.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.burr.Burr.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.burr.Burr.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.burr.Burr.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.burr.Burr.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.burr.Burr.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.burr.Burr.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.burr.Burr.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.burr.Burr.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.burr\_4p module[](#module-phitter.continuous.continuous_distributions.burr_4p "Link to this heading")

*class* phitter.continuous.continuous\_distributions.burr\_4p.Burr4P(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.burr_4p.Burr4P "Link to this definition")
:   Bases: `object`

## Burr distribution
## - Parameters Burr4P Distribution: {“A”: \*, “B”: \*, “C”: \*, “loc”: \*}
## - <https://phitter.io/distributions/continuous/burr_4p>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.burr_4p.Burr4P.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.burr_4p.Burr4P.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.burr_4p.Burr4P.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“A”: \*, “B”: \*, “C”: \*, “loc”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.burr_4p.Burr4P.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.burr_4p.Burr4P.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.burr_4p.Burr4P.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.burr_4p.Burr4P.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.burr_4p.Burr4P.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.burr_4p.Burr4P.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.burr_4p.Burr4P.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.burr_4p.Burr4P.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.burr_4p.Burr4P.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.burr_4p.Burr4P.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.burr_4p.Burr4P.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.burr_4p.Burr4P.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.burr_4p.Burr4P.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.burr_4p.Burr4P.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.burr_4p.Burr4P.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.cauchy module[](#module-phitter.continuous.continuous_distributions.cauchy "Link to this heading")

*class* phitter.continuous.continuous\_distributions.cauchy.Cauchy(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.cauchy.Cauchy "Link to this definition")
:   Bases: `object`

## Cauchy distribution
## - Parameters Cauchy Distribution: {“x0”: \*, “gamma”: \*}
## - <https://phitter.io/distributions/continuous/cauchy>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.cauchy.Cauchy.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.cauchy.Cauchy.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.cauchy.Cauchy.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“x0”: \*, “gamma”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.cauchy.Cauchy.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.cauchy.Cauchy.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.cauchy.Cauchy.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.cauchy.Cauchy.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.cauchy.Cauchy.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.cauchy.Cauchy.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.cauchy.Cauchy.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.cauchy.Cauchy.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.cauchy.Cauchy.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.cauchy.Cauchy.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.cauchy.Cauchy.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.cauchy.Cauchy.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.cauchy.Cauchy.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.cauchy.Cauchy.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.cauchy.Cauchy.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.chi\_square module[](#module-phitter.continuous.continuous_distributions.chi_square "Link to this heading")

*class* phitter.continuous.continuous\_distributions.chi\_square.ChiSquare(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.chi_square.ChiSquare "Link to this definition")
:   Bases: `object`

## Chi Square distribution
## - Parameters ChiSquare Distribution: {“df”: \*}
## - <https://phitter.io/distributions/continuous/chi_square>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.chi_square.ChiSquare.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.chi_square.ChiSquare.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.chi_square.ChiSquare.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“df”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.chi_square.ChiSquare.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.chi_square.ChiSquare.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.chi_square.ChiSquare.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.chi_square.ChiSquare.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.chi_square.ChiSquare.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.chi_square.ChiSquare.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.chi_square.ChiSquare.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.chi_square.ChiSquare.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.chi_square.ChiSquare.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.chi_square.ChiSquare.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.chi_square.ChiSquare.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.chi_square.ChiSquare.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.chi_square.ChiSquare.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.chi_square.ChiSquare.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.chi_square.ChiSquare.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.chi\_square\_3p module[](#module-phitter.continuous.continuous_distributions.chi_square_3p "Link to this heading")

*class* phitter.continuous.continuous\_distributions.chi\_square\_3p.ChiSquare3P(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.chi_square_3p.ChiSquare3P "Link to this definition")
:   Bases: `object`

## Chi Square distribution
## - Parameters ChiSquare3P Distribution: {“df”: \*, “loc”: \*, “scale”: \*}
## - <https://phitter.io/distributions/continuous/chi_square_3p>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.chi_square_3p.ChiSquare3P.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.chi_square_3p.ChiSquare3P.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.chi_square_3p.ChiSquare3P.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“df”: \*, “loc”: \*, “scale”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.chi_square_3p.ChiSquare3P.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.chi_square_3p.ChiSquare3P.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.chi_square_3p.ChiSquare3P.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.chi_square_3p.ChiSquare3P.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.chi_square_3p.ChiSquare3P.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.chi_square_3p.ChiSquare3P.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.chi_square_3p.ChiSquare3P.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.chi_square_3p.ChiSquare3P.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.chi_square_3p.ChiSquare3P.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.chi_square_3p.ChiSquare3P.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.chi_square_3p.ChiSquare3P.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.chi_square_3p.ChiSquare3P.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.chi_square_3p.ChiSquare3P.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.chi_square_3p.ChiSquare3P.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.chi_square_3p.ChiSquare3P.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.dagum module[](#module-phitter.continuous.continuous_distributions.dagum "Link to this heading")

*class* phitter.continuous.continuous\_distributions.dagum.Dagum(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.dagum.Dagum "Link to this definition")
:   Bases: `object`

## Dagum distribution
## - Parameters Dagum Distribution: {“a”: \*, “b”: \*, “p”: \*}
## - <https://phitter.io/distributions/continuous/dagum>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.dagum.Dagum.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.dagum.Dagum.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.dagum.Dagum.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“a”: \*, “b”: \*, “p”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.dagum.Dagum.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.dagum.Dagum.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.dagum.Dagum.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.dagum.Dagum.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.dagum.Dagum.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.dagum.Dagum.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.dagum.Dagum.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.dagum.Dagum.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.dagum.Dagum.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.dagum.Dagum.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.dagum.Dagum.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.dagum.Dagum.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.dagum.Dagum.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.dagum.Dagum.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.dagum.Dagum.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.dagum\_4p module[](#module-phitter.continuous.continuous_distributions.dagum_4p "Link to this heading")

*class* phitter.continuous.continuous\_distributions.dagum\_4p.Dagum4P(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.dagum_4p.Dagum4P "Link to this definition")
:   Bases: `object`

## Dagum distribution
## - Parameters Dagum4P Distribution: {“a”: \*, “b”: \*, “p”: \*, “loc”: \*}
## - <https://phitter.io/distributions/continuous/dagum_4p>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.dagum_4p.Dagum4P.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.dagum_4p.Dagum4P.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.dagum_4p.Dagum4P.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“a”: \*, “b”: \*, “p”: \*, “loc”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.dagum_4p.Dagum4P.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.dagum_4p.Dagum4P.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.dagum_4p.Dagum4P.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.dagum_4p.Dagum4P.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.dagum_4p.Dagum4P.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.dagum_4p.Dagum4P.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.dagum_4p.Dagum4P.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.dagum_4p.Dagum4P.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.dagum_4p.Dagum4P.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.dagum_4p.Dagum4P.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.dagum_4p.Dagum4P.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.dagum_4p.Dagum4P.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.dagum_4p.Dagum4P.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.dagum_4p.Dagum4P.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.dagum_4p.Dagum4P.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.erlang module[](#module-phitter.continuous.continuous_distributions.erlang "Link to this heading")

*class* phitter.continuous.continuous\_distributions.erlang.Erlang(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.erlang.Erlang "Link to this definition")
:   Bases: `object`

## Erlang distribution
## - Parameters Erlang Distribution: {“k”: \*, “beta”: \*}
## - <https://phitter.io/distributions/continuous/erlang>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.erlang.Erlang.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.erlang.Erlang.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.erlang.Erlang.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“k”: \*, “beta”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.erlang.Erlang.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.erlang.Erlang.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.erlang.Erlang.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.erlang.Erlang.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.erlang.Erlang.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.erlang.Erlang.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.erlang.Erlang.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.erlang.Erlang.parameter_restrictions "Link to this definition")
Check parameters restriction

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.erlang.Erlang.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.erlang.Erlang.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.erlang.Erlang.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.erlang.Erlang.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.erlang.Erlang.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.erlang.Erlang.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.erlang.Erlang.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.erlang\_3p module[](#module-phitter.continuous.continuous_distributions.erlang_3p "Link to this heading")

*class* phitter.continuous.continuous\_distributions.erlang\_3p.Erlang3P(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.erlang_3p.Erlang3P "Link to this definition")
:   Bases: `object`

## Erlang 3p distribution
## - Parameters Erlang3P Distribution: {“k”: \*, “beta”: \*, “loc”: \*}
## - <https://phitter.io/distributions/continuous/erlang_3p>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.erlang_3p.Erlang3P.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.erlang_3p.Erlang3P.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.erlang_3p.Erlang3P.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“k”: \*, “beta”: \*, “loc”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.erlang_3p.Erlang3P.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.erlang_3p.Erlang3P.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.erlang_3p.Erlang3P.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.erlang_3p.Erlang3P.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.erlang_3p.Erlang3P.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.erlang_3p.Erlang3P.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.erlang_3p.Erlang3P.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.erlang_3p.Erlang3P.parameter_restrictions "Link to this definition")
Check parameters restriction

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.erlang_3p.Erlang3P.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.erlang_3p.Erlang3P.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.erlang_3p.Erlang3P.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.erlang_3p.Erlang3P.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.erlang_3p.Erlang3P.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.erlang_3p.Erlang3P.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.erlang_3p.Erlang3P.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.error\_function module[](#module-phitter.continuous.continuous_distributions.error_function "Link to this heading")

*class* phitter.continuous.continuous\_distributions.error\_function.ErrorFunction(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.error_function.ErrorFunction "Link to this definition")
:   Bases: `object`

## Error Function distribution
## - Parameters ErrorFunction Distribution: {“h”: \*}
## - <https://phitter.io/distributions/continuous/error_function>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.error_function.ErrorFunction.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.error_function.ErrorFunction.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.error_function.ErrorFunction.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“h”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.error_function.ErrorFunction.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.error_function.ErrorFunction.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.error_function.ErrorFunction.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.error_function.ErrorFunction.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.error_function.ErrorFunction.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.error_function.ErrorFunction.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.error_function.ErrorFunction.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.error_function.ErrorFunction.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.error_function.ErrorFunction.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.error_function.ErrorFunction.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.error_function.ErrorFunction.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.error_function.ErrorFunction.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.error_function.ErrorFunction.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.error_function.ErrorFunction.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.error_function.ErrorFunction.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.exponential module[](#module-phitter.continuous.continuous_distributions.exponential "Link to this heading")

*class* phitter.continuous.continuous\_distributions.exponential.Exponential(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.exponential.Exponential "Link to this definition")
:   Bases: `object`

## Exponential distribution
## - Parameters Exponential Distribution: {“lambda”: \*}
## - <https://phitter.io/distributions/continuous/exponential>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.exponential.Exponential.cdf "Link to this definition")
Cumulative distribution function.
- Calculated with known formula.

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.exponential.Exponential.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.exponential.Exponential.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by solving the equations of the measures expected
- for this distribution.The number of equations to consider is equal to the number
- of parameters.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“lambda”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.exponential.Exponential.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.exponential.Exponential.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.exponential.Exponential.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.exponential.Exponential.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.exponential.Exponential.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.exponential.Exponential.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.exponential.Exponential.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.exponential.Exponential.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.exponential.Exponential.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.exponential.Exponential.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.exponential.Exponential.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.exponential.Exponential.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.exponential.Exponential.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.exponential.Exponential.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.exponential.Exponential.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.exponential\_2p module[](#module-phitter.continuous.continuous_distributions.exponential_2p "Link to this heading")

*class* phitter.continuous.continuous\_distributions.exponential\_2p.Exponential2P(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.exponential_2p.Exponential2P "Link to this definition")
:   Bases: `object`

## Exponential distribution
## - Parameters Exponential2P Distribution: {“lambda”: \*, “loc”: \*}
## - <https://phitter.io/distributions/continuous/exponential_2p>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.exponential_2p.Exponential2P.cdf "Link to this definition")
Cumulative distribution function.
- Calculated with known formula.

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.exponential_2p.Exponential2P.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.exponential_2p.Exponential2P.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by solving the equations of the measures expected
- for this distribution.The number of equations to consider is equal to the number
- of parameters.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“lambda”: \*, “loc”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.exponential_2p.Exponential2P.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.exponential_2p.Exponential2P.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.exponential_2p.Exponential2P.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.exponential_2p.Exponential2P.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.exponential_2p.Exponential2P.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.exponential_2p.Exponential2P.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.exponential_2p.Exponential2P.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.exponential_2p.Exponential2P.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.exponential_2p.Exponential2P.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.exponential_2p.Exponential2P.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.exponential_2p.Exponential2P.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.exponential_2p.Exponential2P.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.exponential_2p.Exponential2P.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.exponential_2p.Exponential2P.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.exponential_2p.Exponential2P.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.f module[](#module-phitter.continuous.continuous_distributions.f "Link to this heading")

*class* phitter.continuous.continuous\_distributions.f.F(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.f.F "Link to this definition")
:   Bases: `object`

## F distribution
## - Parameters F Distribution: {“df1”: \*, “df2”: \*}
## - <https://phitter.io/distributions/continuous/f>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.f.F.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.f.F.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.f.F.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“df1”: \*, “df2”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.f.F.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.f.F.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.f.F.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.f.F.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.f.F.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.f.F.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.f.F.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.f.F.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.f.F.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.f.F.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.f.F.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.f.F.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.f.F.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.f.F.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.f.F.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.f\_4p module[](#module-phitter.continuous.continuous_distributions.f_4p "Link to this heading")

*class* phitter.continuous.continuous\_distributions.f\_4p.F4P(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.f_4p.F4P "Link to this definition")
:   Bases: `object`

## F distribution
## - Parameters F4P Distribution: {“df1”: \*, “df2”: \*, “loc”: \*, “scale”: \*}
## - <https://phitter.io/distributions/continuous/f_4p>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.f_4p.F4P.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.f_4p.F4P.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.f_4p.F4P.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“df1”: \*, “df2”: \*, “loc”: \*, “scale”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.f_4p.F4P.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.f_4p.F4P.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.f_4p.F4P.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.f_4p.F4P.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.f_4p.F4P.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.f_4p.F4P.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.f_4p.F4P.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.f_4p.F4P.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.f_4p.F4P.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.f_4p.F4P.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.f_4p.F4P.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.f_4p.F4P.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.f_4p.F4P.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.f_4p.F4P.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.f_4p.F4P.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.fatigue\_life module[](#module-phitter.continuous.continuous_distributions.fatigue_life "Link to this heading")

*class* phitter.continuous.continuous\_distributions.fatigue\_life.FatigueLife(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.fatigue_life.FatigueLife "Link to this definition")
:   Bases: `object`

## Fatigue life Distribution
## Also known as Birnbaum-Saunders distribution
## - Parameters FatigueLife Distribution: {“gamma”: \*, “loc”: \*, “scale”: \*}
## - <https://phitter.io/distributions/continuous/fatigue_life>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.fatigue_life.FatigueLife.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.fatigue_life.FatigueLife.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.fatigue_life.FatigueLife.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“gamma”: \*, “loc”: \*, “scale”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.fatigue_life.FatigueLife.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.fatigue_life.FatigueLife.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.fatigue_life.FatigueLife.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.fatigue_life.FatigueLife.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.fatigue_life.FatigueLife.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.fatigue_life.FatigueLife.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.fatigue_life.FatigueLife.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.fatigue_life.FatigueLife.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.fatigue_life.FatigueLife.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.fatigue_life.FatigueLife.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.fatigue_life.FatigueLife.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.fatigue_life.FatigueLife.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.fatigue_life.FatigueLife.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.fatigue_life.FatigueLife.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.fatigue_life.FatigueLife.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.folded\_normal module[](#module-phitter.continuous.continuous_distributions.folded_normal "Link to this heading")

*class* phitter.continuous.continuous\_distributions.folded\_normal.FoldedNormal(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.folded_normal.FoldedNormal "Link to this definition")
:   Bases: `object`

## Folded Normal Distribution
## - <https://phitter.io/distributions/continuous/folded_normal>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.folded_normal.FoldedNormal.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.folded_normal.FoldedNormal.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.folded_normal.FoldedNormal.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*dict*) – {“mu”: \* , “variance”: \* , “skewness”: \* , “kurtosis”: \* , “data”: \* }

#### Returns
**parameters**

#### Return type
{“mu”: \*, “sigma”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.folded_normal.FoldedNormal.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.folded_normal.FoldedNormal.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.folded_normal.FoldedNormal.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.folded_normal.FoldedNormal.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.folded_normal.FoldedNormal.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.folded_normal.FoldedNormal.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.folded_normal.FoldedNormal.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.folded_normal.FoldedNormal.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.folded_normal.FoldedNormal.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.folded_normal.FoldedNormal.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.folded_normal.FoldedNormal.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.folded_normal.FoldedNormal.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.folded_normal.FoldedNormal.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.folded_normal.FoldedNormal.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.folded_normal.FoldedNormal.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.frechet module[](#module-phitter.continuous.continuous_distributions.frechet "Link to this heading")

*class* phitter.continuous.continuous\_distributions.frechet.Frechet(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.frechet.Frechet "Link to this definition")
:   Bases: `object`

## Fréchet Distribution
## Also known as inverse Weibull distribution (Scipy name)
## - <https://phitter.io/distributions/continuous/frechet>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.frechet.Frechet.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.frechet.Frechet.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.frechet.Frechet.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“alpha”: \*, “loc”: \*, “scale”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.frechet.Frechet.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.frechet.Frechet.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.frechet.Frechet.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.frechet.Frechet.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.frechet.Frechet.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.frechet.Frechet.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.frechet.Frechet.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.frechet.Frechet.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.frechet.Frechet.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.frechet.Frechet.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.frechet.Frechet.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.frechet.Frechet.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.frechet.Frechet.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.frechet.Frechet.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.frechet.Frechet.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.gamma module[](#module-phitter.continuous.continuous_distributions.gamma "Link to this heading")

*class* phitter.continuous.continuous\_distributions.gamma.Gamma(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.gamma.Gamma "Link to this definition")
:   Bases: `object`

## Gamma distribution
## - Parameters Gamma Distribution: {“alpha”: \*, “beta”: \*}
## - <https://phitter.io/distributions/continuous/gamma>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.gamma.Gamma.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.gamma.Gamma.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.gamma.Gamma.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“alpha”: \*, “beta”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.gamma.Gamma.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.gamma.Gamma.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.gamma.Gamma.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.gamma.Gamma.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.gamma.Gamma.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.gamma.Gamma.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.gamma.Gamma.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.gamma.Gamma.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.gamma.Gamma.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.gamma.Gamma.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.gamma.Gamma.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.gamma.Gamma.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.gamma.Gamma.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.gamma.Gamma.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.gamma.Gamma.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.gamma\_3p module[](#module-phitter.continuous.continuous_distributions.gamma_3p "Link to this heading")

*class* phitter.continuous.continuous\_distributions.gamma\_3p.Gamma3P(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.gamma_3p.Gamma3P "Link to this definition")
:   Bases: `object`

## Gamma distribution
## - Parameters Gamma3P Distribution: {“alpha”: \*, “loc”: \*, “beta”: \*}
## - <https://phitter.io/distributions/continuous/gamma_3p>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.gamma_3p.Gamma3P.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.gamma_3p.Gamma3P.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.gamma_3p.Gamma3P.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“alpha”: \*, “loc”: \*, “beta”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.gamma_3p.Gamma3P.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.gamma_3p.Gamma3P.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.gamma_3p.Gamma3P.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.gamma_3p.Gamma3P.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.gamma_3p.Gamma3P.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.gamma_3p.Gamma3P.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.gamma_3p.Gamma3P.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.gamma_3p.Gamma3P.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.gamma_3p.Gamma3P.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.gamma_3p.Gamma3P.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.gamma_3p.Gamma3P.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.gamma_3p.Gamma3P.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.gamma_3p.Gamma3P.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.gamma_3p.Gamma3P.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.gamma_3p.Gamma3P.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.generalized\_extreme\_value module[](#module-phitter.continuous.continuous_distributions.generalized_extreme_value "Link to this heading")

*class* phitter.continuous.continuous\_distributions.generalized\_extreme\_value.GeneralizedExtremeValue(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.generalized_extreme_value.GeneralizedExtremeValue "Link to this definition")
:   Bases: `object`

## Generalized Extreme Value Distribution
## - <https://phitter.io/distributions/continuous/generalized_extreme_value>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.generalized_extreme_value.GeneralizedExtremeValue.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.generalized_extreme_value.GeneralizedExtremeValue.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.generalized_extreme_value.GeneralizedExtremeValue.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“xi”: \*, “mu”: \*, “sigma”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.generalized_extreme_value.GeneralizedExtremeValue.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.generalized_extreme_value.GeneralizedExtremeValue.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.generalized_extreme_value.GeneralizedExtremeValue.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.generalized_extreme_value.GeneralizedExtremeValue.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.generalized_extreme_value.GeneralizedExtremeValue.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.generalized_extreme_value.GeneralizedExtremeValue.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.generalized_extreme_value.GeneralizedExtremeValue.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.generalized_extreme_value.GeneralizedExtremeValue.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.generalized_extreme_value.GeneralizedExtremeValue.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.generalized_extreme_value.GeneralizedExtremeValue.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.generalized_extreme_value.GeneralizedExtremeValue.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.generalized_extreme_value.GeneralizedExtremeValue.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.generalized_extreme_value.GeneralizedExtremeValue.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.generalized_extreme_value.GeneralizedExtremeValue.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.generalized_extreme_value.GeneralizedExtremeValue.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.generalized\_gamma module[](#module-phitter.continuous.continuous_distributions.generalized_gamma "Link to this heading")

*class* phitter.continuous.continuous\_distributions.generalized\_gamma.GeneralizedGamma(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.generalized_gamma.GeneralizedGamma "Link to this definition")
:   Bases: `object`

## Generalized Gamma Distribution
## - <https://phitter.io/distributions/continuous/generalized_gamma>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.generalized_gamma.GeneralizedGamma.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.generalized_gamma.GeneralizedGamma.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.generalized_gamma.GeneralizedGamma.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“a”: \*, “d”: \*, “p”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.generalized_gamma.GeneralizedGamma.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.generalized_gamma.GeneralizedGamma.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.generalized_gamma.GeneralizedGamma.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.generalized_gamma.GeneralizedGamma.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.generalized_gamma.GeneralizedGamma.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.generalized_gamma.GeneralizedGamma.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.generalized_gamma.GeneralizedGamma.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.generalized_gamma.GeneralizedGamma.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.generalized_gamma.GeneralizedGamma.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.generalized_gamma.GeneralizedGamma.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.generalized_gamma.GeneralizedGamma.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.generalized_gamma.GeneralizedGamma.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.generalized_gamma.GeneralizedGamma.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.generalized_gamma.GeneralizedGamma.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.generalized_gamma.GeneralizedGamma.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.generalized\_gamma\_4p module[](#module-phitter.continuous.continuous_distributions.generalized_gamma_4p "Link to this heading")

*class* phitter.continuous.continuous\_distributions.generalized\_gamma\_4p.GeneralizedGamma4P(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.generalized_gamma_4p.GeneralizedGamma4P "Link to this definition")
:   Bases: `object`

## Generalized Gamma Distribution
## - <https://phitter.io/distributions/continuous/generalized_gamma_4p>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.generalized_gamma_4p.GeneralizedGamma4P.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.generalized_gamma_4p.GeneralizedGamma4P.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.generalized_gamma_4p.GeneralizedGamma4P.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“a”: \*, “d”: \*, “p”: \*, “loc”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.generalized_gamma_4p.GeneralizedGamma4P.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.generalized_gamma_4p.GeneralizedGamma4P.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.generalized_gamma_4p.GeneralizedGamma4P.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.generalized_gamma_4p.GeneralizedGamma4P.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.generalized_gamma_4p.GeneralizedGamma4P.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.generalized_gamma_4p.GeneralizedGamma4P.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.generalized_gamma_4p.GeneralizedGamma4P.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.generalized_gamma_4p.GeneralizedGamma4P.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.generalized_gamma_4p.GeneralizedGamma4P.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.generalized_gamma_4p.GeneralizedGamma4P.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.generalized_gamma_4p.GeneralizedGamma4P.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.generalized_gamma_4p.GeneralizedGamma4P.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.generalized_gamma_4p.GeneralizedGamma4P.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.generalized_gamma_4p.GeneralizedGamma4P.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.generalized_gamma_4p.GeneralizedGamma4P.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.generalized\_logistic module[](#module-phitter.continuous.continuous_distributions.generalized_logistic "Link to this heading")

*class* phitter.continuous.continuous\_distributions.generalized\_logistic.GeneralizedLogistic(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.generalized_logistic.GeneralizedLogistic "Link to this definition")
:   Bases: `object`

## Generalized Logistic Distribution

## * <https://phitter.io/distributions/continuous/generalized_logistic>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.generalized_logistic.GeneralizedLogistic.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.generalized_logistic.GeneralizedLogistic.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.generalized_logistic.GeneralizedLogistic.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“loc”: \*, “scale”: \*, “c”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.generalized_logistic.GeneralizedLogistic.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.generalized_logistic.GeneralizedLogistic.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.generalized_logistic.GeneralizedLogistic.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.generalized_logistic.GeneralizedLogistic.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.generalized_logistic.GeneralizedLogistic.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.generalized_logistic.GeneralizedLogistic.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.generalized_logistic.GeneralizedLogistic.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.generalized_logistic.GeneralizedLogistic.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.generalized_logistic.GeneralizedLogistic.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.generalized_logistic.GeneralizedLogistic.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.generalized_logistic.GeneralizedLogistic.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.generalized_logistic.GeneralizedLogistic.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.generalized_logistic.GeneralizedLogistic.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.generalized_logistic.GeneralizedLogistic.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.generalized_logistic.GeneralizedLogistic.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.generalized\_normal module[](#module-phitter.continuous.continuous_distributions.generalized_normal "Link to this heading")

*class* phitter.continuous.continuous\_distributions.generalized\_normal.GeneralizedNormal(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.generalized_normal.GeneralizedNormal "Link to this definition")
:   Bases: `object`

## Generalized normal distribution
## - Parameters GeneralizedNormal Distribution: {“beta”: \*, “mu”: \*, “alpha”: \*}
## - <https://phitter.io/distributions/continuous/generalized_normal> This distribution is known whit the following names:
## \* Error Distribution
## \* Exponential Power Distribution
## \* Generalized Error Distribution (GED)
## \* Generalized Gaussian distribution (GGD)
## \* Subbotin distribution

## cdf(*x*)[](#phitter.continuous.continuous_distributions.generalized_normal.GeneralizedNormal.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.generalized_normal.GeneralizedNormal.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.generalized_normal.GeneralizedNormal.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“beta”: \*, “mu”: \*, “alpha”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.generalized_normal.GeneralizedNormal.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.generalized_normal.GeneralizedNormal.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.generalized_normal.GeneralizedNormal.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.generalized_normal.GeneralizedNormal.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.generalized_normal.GeneralizedNormal.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.generalized_normal.GeneralizedNormal.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.generalized_normal.GeneralizedNormal.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.generalized_normal.GeneralizedNormal.parameter_restrictions "Link to this definition")
Check parameters restriction

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.generalized_normal.GeneralizedNormal.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.generalized_normal.GeneralizedNormal.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.generalized_normal.GeneralizedNormal.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.generalized_normal.GeneralizedNormal.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.generalized_normal.GeneralizedNormal.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.generalized_normal.GeneralizedNormal.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.generalized_normal.GeneralizedNormal.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.generalized\_pareto module[](#module-phitter.continuous.continuous_distributions.generalized_pareto "Link to this heading")

*class* phitter.continuous.continuous\_distributions.generalized\_pareto.GeneralizedPareto(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.generalized_pareto.GeneralizedPareto "Link to this definition")
:   Bases: `object`

## Generalized Pareto distribution
## - Parameters GeneralizedPareto Distribution: {“c”: \*, “mu”: \*, “sigma”: \*}
## - <https://phitter.io/distributions/continuous/generalized_pareto>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.generalized_pareto.GeneralizedPareto.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.generalized_pareto.GeneralizedPareto.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.generalized_pareto.GeneralizedPareto.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“c”: \*, “mu”: \*, “sigma”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.generalized_pareto.GeneralizedPareto.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.generalized_pareto.GeneralizedPareto.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.generalized_pareto.GeneralizedPareto.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.generalized_pareto.GeneralizedPareto.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.generalized_pareto.GeneralizedPareto.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.generalized_pareto.GeneralizedPareto.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.generalized_pareto.GeneralizedPareto.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.generalized_pareto.GeneralizedPareto.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.generalized_pareto.GeneralizedPareto.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.generalized_pareto.GeneralizedPareto.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.generalized_pareto.GeneralizedPareto.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.generalized_pareto.GeneralizedPareto.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.generalized_pareto.GeneralizedPareto.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.generalized_pareto.GeneralizedPareto.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.generalized_pareto.GeneralizedPareto.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.gibrat module[](#module-phitter.continuous.continuous_distributions.gibrat "Link to this heading")

*class* phitter.continuous.continuous\_distributions.gibrat.Gibrat(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.gibrat.Gibrat "Link to this definition")
:   Bases: `object`

## Gibrat distribution
## - Parameters Gibrat Distribution: {“loc”: \*, “scale”: \*}
## - <https://phitter.io/distributions/continuous/gibrat>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.gibrat.Gibrat.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.gibrat.Gibrat.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.gibrat.Gibrat.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by solving the equations of the measures expected
- for this distribution.The number of equations to consider is equal to the number
- of parameters.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“loc”: \*, “scale”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.gibrat.Gibrat.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.gibrat.Gibrat.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.gibrat.Gibrat.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.gibrat.Gibrat.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.gibrat.Gibrat.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.gibrat.Gibrat.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.gibrat.Gibrat.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.gibrat.Gibrat.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.gibrat.Gibrat.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.gibrat.Gibrat.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.gibrat.Gibrat.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.gibrat.Gibrat.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.gibrat.Gibrat.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.gibrat.Gibrat.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.gibrat.Gibrat.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.gumbel\_left module[](#module-phitter.continuous.continuous_distributions.gumbel_left "Link to this heading")

*class* phitter.continuous.continuous\_distributions.gumbel\_left.GumbelLeft(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.gumbel_left.GumbelLeft "Link to this definition")
:   Bases: `object`

## Gumbel Left Distribution
## Gumbel Min Distribution
## Extreme Value Minimum Distribution
## - <https://phitter.io/distributions/continuous/gumbel_left>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.gumbel_left.GumbelLeft.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.gumbel_left.GumbelLeft.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.gumbel_left.GumbelLeft.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“mu”: \*, “sigma”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.gumbel_left.GumbelLeft.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.gumbel_left.GumbelLeft.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.gumbel_left.GumbelLeft.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.gumbel_left.GumbelLeft.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.gumbel_left.GumbelLeft.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.gumbel_left.GumbelLeft.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.gumbel_left.GumbelLeft.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.gumbel_left.GumbelLeft.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.gumbel_left.GumbelLeft.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.gumbel_left.GumbelLeft.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.gumbel_left.GumbelLeft.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.gumbel_left.GumbelLeft.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.gumbel_left.GumbelLeft.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.gumbel_left.GumbelLeft.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.gumbel_left.GumbelLeft.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.gumbel\_right module[](#module-phitter.continuous.continuous_distributions.gumbel_right "Link to this heading")

*class* phitter.continuous.continuous\_distributions.gumbel\_right.GumbelRight(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.gumbel_right.GumbelRight "Link to this definition")
:   Bases: `object`

## Gumbel Right Distribution
## Gumbel Max Distribution
## Extreme Value Maximum Distribution

## * <https://phitter.io/distributions/continuous/gumbel_right>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.gumbel_right.GumbelRight.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.gumbel_right.GumbelRight.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.gumbel_right.GumbelRight.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“mu”: \*, “sigma”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.gumbel_right.GumbelRight.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.gumbel_right.GumbelRight.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.gumbel_right.GumbelRight.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.gumbel_right.GumbelRight.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.gumbel_right.GumbelRight.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.gumbel_right.GumbelRight.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.gumbel_right.GumbelRight.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.gumbel_right.GumbelRight.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.gumbel_right.GumbelRight.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.gumbel_right.GumbelRight.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.gumbel_right.GumbelRight.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.gumbel_right.GumbelRight.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.gumbel_right.GumbelRight.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.gumbel_right.GumbelRight.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.gumbel_right.GumbelRight.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.half\_normal module[](#module-phitter.continuous.continuous_distributions.half_normal "Link to this heading")

*class* phitter.continuous.continuous\_distributions.half\_normal.HalfNormal(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.half_normal.HalfNormal "Link to this definition")
:   Bases: `object`

## Half Normal Distribution
## - <https://phitter.io/distributions/continuous/half_normal>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.half_normal.HalfNormal.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.half_normal.HalfNormal.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.half_normal.HalfNormal.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*dict*) – {“mu”: \* , “variance”: \* , “skewness”: \* , “kurtosis”: \* , “data”: \* }

#### Returns
**parameters**

#### Return type
{“mu”: \*, “sigma”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.half_normal.HalfNormal.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.half_normal.HalfNormal.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.half_normal.HalfNormal.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.half_normal.HalfNormal.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.half_normal.HalfNormal.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.half_normal.HalfNormal.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.half_normal.HalfNormal.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.half_normal.HalfNormal.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.half_normal.HalfNormal.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.half_normal.HalfNormal.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.half_normal.HalfNormal.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.half_normal.HalfNormal.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.half_normal.HalfNormal.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.half_normal.HalfNormal.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.half_normal.HalfNormal.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.hyperbolic\_secant module[](#module-phitter.continuous.continuous_distributions.hyperbolic_secant "Link to this heading")

*class* phitter.continuous.continuous\_distributions.hyperbolic\_secant.HyperbolicSecant(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.hyperbolic_secant.HyperbolicSecant "Link to this definition")
:   Bases: `object`

## Hyperbolic Secant distribution
## - Parameters HyperbolicSecant Distribution: {“mu”: \*, “sigma”: \*}
## - <https://phitter.io/distributions/continuous/hyperbolic_secant>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.hyperbolic_secant.HyperbolicSecant.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.hyperbolic_secant.HyperbolicSecant.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.hyperbolic_secant.HyperbolicSecant.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*dict*) – {“mu”: \* , “variance”: \* , “skewness”: \* , “kurtosis”: \* , “data”: \* }

#### Returns
**parameters**

#### Return type
{“mu”: \*, “sigma”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.hyperbolic_secant.HyperbolicSecant.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.hyperbolic_secant.HyperbolicSecant.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.hyperbolic_secant.HyperbolicSecant.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.hyperbolic_secant.HyperbolicSecant.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.hyperbolic_secant.HyperbolicSecant.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.hyperbolic_secant.HyperbolicSecant.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.hyperbolic_secant.HyperbolicSecant.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.hyperbolic_secant.HyperbolicSecant.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.hyperbolic_secant.HyperbolicSecant.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.hyperbolic_secant.HyperbolicSecant.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.hyperbolic_secant.HyperbolicSecant.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.hyperbolic_secant.HyperbolicSecant.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.hyperbolic_secant.HyperbolicSecant.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.hyperbolic_secant.HyperbolicSecant.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.hyperbolic_secant.HyperbolicSecant.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.inverse\_gamma module[](#module-phitter.continuous.continuous_distributions.inverse_gamma "Link to this heading")

*class* phitter.continuous.continuous\_distributions.inverse\_gamma.InverseGamma(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.inverse_gamma.InverseGamma "Link to this definition")
:   Bases: `object`

## Inverse Gamma Distribution
## Also known Pearson Type 5 distribution
## - Parameters InverseGamma Distribution: {“alpha”: \*, “beta”: \*}
## - <https://phitter.io/distributions/continuous/inverse_gamma>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.inverse_gamma.InverseGamma.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.inverse_gamma.InverseGamma.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.inverse_gamma.InverseGamma.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“alpha”: \*, “beta”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.inverse_gamma.InverseGamma.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.inverse_gamma.InverseGamma.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.inverse_gamma.InverseGamma.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.inverse_gamma.InverseGamma.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.inverse_gamma.InverseGamma.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.inverse_gamma.InverseGamma.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.inverse_gamma.InverseGamma.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.inverse_gamma.InverseGamma.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.inverse_gamma.InverseGamma.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.inverse_gamma.InverseGamma.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.inverse_gamma.InverseGamma.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.inverse_gamma.InverseGamma.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.inverse_gamma.InverseGamma.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.inverse_gamma.InverseGamma.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.inverse_gamma.InverseGamma.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.inverse\_gamma\_3p module[](#module-phitter.continuous.continuous_distributions.inverse_gamma_3p "Link to this heading")

*class* phitter.continuous.continuous\_distributions.inverse\_gamma\_3p.InverseGamma3P(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.inverse_gamma_3p.InverseGamma3P "Link to this definition")
:   Bases: `object`

## Inverse Gamma Distribution
## Also known Pearson Type 5 distribution
## - Parameters InverseGamma3P Distribution: {“alpha”: \*, “beta”: \*, “loc”: \*}
## - <https://phitter.io/distributions/continuous/inverse_gamma_3p>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.inverse_gamma_3p.InverseGamma3P.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.inverse_gamma_3p.InverseGamma3P.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.inverse_gamma_3p.InverseGamma3P.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“alpha”: \*, “beta”: \*, “loc”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.inverse_gamma_3p.InverseGamma3P.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.inverse_gamma_3p.InverseGamma3P.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.inverse_gamma_3p.InverseGamma3P.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.inverse_gamma_3p.InverseGamma3P.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.inverse_gamma_3p.InverseGamma3P.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.inverse_gamma_3p.InverseGamma3P.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.inverse_gamma_3p.InverseGamma3P.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.inverse_gamma_3p.InverseGamma3P.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.inverse_gamma_3p.InverseGamma3P.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.inverse_gamma_3p.InverseGamma3P.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.inverse_gamma_3p.InverseGamma3P.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.inverse_gamma_3p.InverseGamma3P.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.inverse_gamma_3p.InverseGamma3P.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.inverse_gamma_3p.InverseGamma3P.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.inverse_gamma_3p.InverseGamma3P.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.inverse\_gaussian module[](#module-phitter.continuous.continuous_distributions.inverse_gaussian "Link to this heading")

*class* phitter.continuous.continuous\_distributions.inverse\_gaussian.InverseGaussian(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.inverse_gaussian.InverseGaussian "Link to this definition")
:   Bases: `object`

## Inverse Gaussian Distribution
## Also known like Wald distribution
## - Parameters InverseGaussian Distribution: {“mu”: \*, “lambda”: \*}
## - <https://phitter.io/distributions/continuous/inverse_gaussian>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.inverse_gaussian.InverseGaussian.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.inverse_gaussian.InverseGaussian.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.inverse_gaussian.InverseGaussian.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*dict*) – {“mu”: \* , “variance”: \* , “skewness”: \* , “kurtosis”: \* , “data”: \* }

#### Returns
**parameters**

#### Return type
{“mu”: \*, “lambda”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.inverse_gaussian.InverseGaussian.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.inverse_gaussian.InverseGaussian.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.inverse_gaussian.InverseGaussian.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.inverse_gaussian.InverseGaussian.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.inverse_gaussian.InverseGaussian.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.inverse_gaussian.InverseGaussian.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.inverse_gaussian.InverseGaussian.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.inverse_gaussian.InverseGaussian.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.inverse_gaussian.InverseGaussian.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.inverse_gaussian.InverseGaussian.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.inverse_gaussian.InverseGaussian.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.inverse_gaussian.InverseGaussian.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.inverse_gaussian.InverseGaussian.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.inverse_gaussian.InverseGaussian.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.inverse_gaussian.InverseGaussian.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.inverse\_gaussian\_3p module[](#module-phitter.continuous.continuous_distributions.inverse_gaussian_3p "Link to this heading")

*class* phitter.continuous.continuous\_distributions.inverse\_gaussian\_3p.InverseGaussian3P(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.inverse_gaussian_3p.InverseGaussian3P "Link to this definition")
:   Bases: `object`

## Inverse Gaussian Distribution
## Also known like Wald distribution
## - Parameters InverseGaussian3P Distribution: {“mu”: \*, “lambda”: \*, “loc”: \*}
## - <https://phitter.io/distributions/continuous/inverse_gaussian_3p>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.inverse_gaussian_3p.InverseGaussian3P.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.inverse_gaussian_3p.InverseGaussian3P.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.inverse_gaussian_3p.InverseGaussian3P.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*dict*) – {“mu”: \* , “variance”: \* , “skewness”: \* , “kurtosis”: \* , “data”: \* }

#### Returns
**parameters**

#### Return type
{“mu”: \*, “lambda”: \*, “loc”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.inverse_gaussian_3p.InverseGaussian3P.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.inverse_gaussian_3p.InverseGaussian3P.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.inverse_gaussian_3p.InverseGaussian3P.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.inverse_gaussian_3p.InverseGaussian3P.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.inverse_gaussian_3p.InverseGaussian3P.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.inverse_gaussian_3p.InverseGaussian3P.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.inverse_gaussian_3p.InverseGaussian3P.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.inverse_gaussian_3p.InverseGaussian3P.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.inverse_gaussian_3p.InverseGaussian3P.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.inverse_gaussian_3p.InverseGaussian3P.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.inverse_gaussian_3p.InverseGaussian3P.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.inverse_gaussian_3p.InverseGaussian3P.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.inverse_gaussian_3p.InverseGaussian3P.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.inverse_gaussian_3p.InverseGaussian3P.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.inverse_gaussian_3p.InverseGaussian3P.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.johnson\_sb module[](#module-phitter.continuous.continuous_distributions.johnson_sb "Link to this heading")

*class* phitter.continuous.continuous\_distributions.johnson\_sb.JohnsonSB(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.johnson_sb.JohnsonSB "Link to this definition")
:   Bases: `object`

## Johnson SB distribution
## - Parameters JohnsonSB Distribution: {“xi”: \*, “lambda”: \*, “gamma”: \*, “delta”: \*}
## - <https://phitter.io/distributions/continuous/johnson_sb>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.johnson_sb.JohnsonSB.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.johnson_sb.JohnsonSB.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.johnson_sb.JohnsonSB.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated with the method proposed in [1].

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters** – {“xi”: \* , “lambda”: \* , “gamma”: \* , “delta”: \* }

#### Return type
{“xi”: \*, “lambda”: \*, “gamma”: \*, “delta”: \*}

- References

- [1]

- George, F., & Ramachandran, K. M. (2011).
- Estimation of parameters of Johnson’s system of distributions.
- Journal of Modern Applied Statistical Methods, 10(2), 9.

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.johnson_sb.JohnsonSB.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.johnson_sb.JohnsonSB.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.johnson_sb.JohnsonSB.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.johnson_sb.JohnsonSB.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.johnson_sb.JohnsonSB.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.johnson_sb.JohnsonSB.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.johnson_sb.JohnsonSB.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.johnson_sb.JohnsonSB.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.johnson_sb.JohnsonSB.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.johnson_sb.JohnsonSB.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.johnson_sb.JohnsonSB.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.johnson_sb.JohnsonSB.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.johnson_sb.JohnsonSB.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.johnson_sb.JohnsonSB.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.johnson_sb.JohnsonSB.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.johnson\_su module[](#module-phitter.continuous.continuous_distributions.johnson_su "Link to this heading")

*class* phitter.continuous.continuous\_distributions.johnson\_su.JohnsonSU(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.johnson_su.JohnsonSU "Link to this definition")
:   Bases: `object`

## Johnson SU distribution
## - Parameters JohnsonSU Distribution: {“xi”: \*, “lambda”: \*, “gamma”: \*, “delta”: \*}
## - <https://phitter.io/distributions/continuous/johnson_su>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.johnson_su.JohnsonSU.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.johnson_su.JohnsonSU.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.johnson_su.JohnsonSU.get_parameters "Link to this definition")
Return type
`dict`[`str`, `float` | `int`]

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.johnson_su.JohnsonSU.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.johnson_su.JohnsonSU.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.johnson_su.JohnsonSU.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.johnson_su.JohnsonSU.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.johnson_su.JohnsonSU.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.johnson_su.JohnsonSU.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.johnson_su.JohnsonSU.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.johnson_su.JohnsonSU.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.johnson_su.JohnsonSU.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.johnson_su.JohnsonSU.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.johnson_su.JohnsonSU.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.johnson_su.JohnsonSU.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.johnson_su.JohnsonSU.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.johnson_su.JohnsonSU.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.johnson_su.JohnsonSU.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.kumaraswamy module[](#module-phitter.continuous.continuous_distributions.kumaraswamy "Link to this heading")

*class* phitter.continuous.continuous\_distributions.kumaraswamy.Kumaraswamy(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.kumaraswamy.Kumaraswamy "Link to this definition")
:   Bases: `object`

## Kumaraswami distribution
## - Parameters Kumaraswamy Distribution: {“alpha”: \*, “beta”: \*, “min”: \*, “max”: \*}
## - <https://phitter.io/distributions/continuous/kumaraswamy>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.kumaraswamy.Kumaraswamy.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.kumaraswamy.Kumaraswamy.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.kumaraswamy.Kumaraswamy.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by solving the equations of the measures expected
- for this distribution.The number of equations to consider is equal to the number
- of parameters.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“alpha”: \*, “beta”: \*, “min”: \*, “max”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.kumaraswamy.Kumaraswamy.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.kumaraswamy.Kumaraswamy.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.kumaraswamy.Kumaraswamy.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.kumaraswamy.Kumaraswamy.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.kumaraswamy.Kumaraswamy.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.kumaraswamy.Kumaraswamy.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.kumaraswamy.Kumaraswamy.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.kumaraswamy.Kumaraswamy.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.kumaraswamy.Kumaraswamy.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.kumaraswamy.Kumaraswamy.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.kumaraswamy.Kumaraswamy.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.kumaraswamy.Kumaraswamy.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.kumaraswamy.Kumaraswamy.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.kumaraswamy.Kumaraswamy.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.kumaraswamy.Kumaraswamy.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.laplace module[](#module-phitter.continuous.continuous_distributions.laplace "Link to this heading")

*class* phitter.continuous.continuous\_distributions.laplace.Laplace(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.laplace.Laplace "Link to this definition")
:   Bases: `object`

## Laplace distribution
## - Parameters Laplace Distribution: {“mu”: \*, “b”: \*}
## - <https://phitter.io/distributions/continuous/laplace>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.laplace.Laplace.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.laplace.Laplace.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.laplace.Laplace.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“mu”: \*, “b”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.laplace.Laplace.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.laplace.Laplace.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.laplace.Laplace.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.laplace.Laplace.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.laplace.Laplace.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.laplace.Laplace.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.laplace.Laplace.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.laplace.Laplace.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.laplace.Laplace.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.laplace.Laplace.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.laplace.Laplace.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.laplace.Laplace.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.laplace.Laplace.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.laplace.Laplace.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.laplace.Laplace.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.levy module[](#module-phitter.continuous.continuous_distributions.levy "Link to this heading")

*class* phitter.continuous.continuous\_distributions.levy.Levy(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.levy.Levy "Link to this definition")
:   Bases: `object`

## Levy distribution
## - Parameters Levy Distribution: {“mu”: \*, “c”: \*}
## - <https://phitter.io/distributions/continuous/levy>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.levy.Levy.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.levy.Levy.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.levy.Levy.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“mu”: \*, “c”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.levy.Levy.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.levy.Levy.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.levy.Levy.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.levy.Levy.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.levy.Levy.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.levy.Levy.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.levy.Levy.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.levy.Levy.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.levy.Levy.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.levy.Levy.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.levy.Levy.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.levy.Levy.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.levy.Levy.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.levy.Levy.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.levy.Levy.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.loggamma module[](#module-phitter.continuous.continuous_distributions.loggamma "Link to this heading")

*class* phitter.continuous.continuous\_distributions.loggamma.LogGamma(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.loggamma.LogGamma "Link to this definition")
:   Bases: `object`

## LogGamma distribution
## - Parameters LogGamma Distribution: {“c”: \*, “mu”: \*, “sigma”: \*}
## - <https://phitter.io/distributions/continuous/loggamma>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.loggamma.LogGamma.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.loggamma.LogGamma.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.loggamma.LogGamma.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“c”: \*, “mu”: \*, “sigma”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.loggamma.LogGamma.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.loggamma.LogGamma.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.loggamma.LogGamma.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.loggamma.LogGamma.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.loggamma.LogGamma.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.loggamma.LogGamma.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.loggamma.LogGamma.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.loggamma.LogGamma.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.loggamma.LogGamma.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.loggamma.LogGamma.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.loggamma.LogGamma.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.loggamma.LogGamma.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.loggamma.LogGamma.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.loggamma.LogGamma.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.loggamma.LogGamma.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.logistic module[](#module-phitter.continuous.continuous_distributions.logistic "Link to this heading")

*class* phitter.continuous.continuous\_distributions.logistic.Logistic(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.logistic.Logistic "Link to this definition")
:   Bases: `object`

## Logistic distribution
## - Parameters Logistic Distribution: {“mu”: \*, “sigma”: \*}
## - <https://phitter.io/distributions/continuous/logistic>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.logistic.Logistic.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.logistic.Logistic.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.logistic.Logistic.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“mu”: \*, “sigma”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.logistic.Logistic.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.logistic.Logistic.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.logistic.Logistic.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.logistic.Logistic.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.logistic.Logistic.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.logistic.Logistic.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.logistic.Logistic.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.logistic.Logistic.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.logistic.Logistic.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.logistic.Logistic.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.logistic.Logistic.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.logistic.Logistic.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.logistic.Logistic.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.logistic.Logistic.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.logistic.Logistic.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.loglogistic module[](#module-phitter.continuous.continuous_distributions.loglogistic "Link to this heading")

*class* phitter.continuous.continuous\_distributions.loglogistic.LogLogistic(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.loglogistic.LogLogistic "Link to this definition")
:   Bases: `object`

## Loglogistic distribution
## - Parameters LogLogistic Distribution: {“alpha”: \*, “beta”: \*}
## - <https://phitter.io/distributions/continuous/loglogistic>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.loglogistic.LogLogistic.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.loglogistic.LogLogistic.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.loglogistic.LogLogistic.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“alpha”: \*, “beta”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.loglogistic.LogLogistic.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.loglogistic.LogLogistic.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.loglogistic.LogLogistic.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.loglogistic.LogLogistic.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.loglogistic.LogLogistic.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.loglogistic.LogLogistic.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.loglogistic.LogLogistic.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.loglogistic.LogLogistic.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.loglogistic.LogLogistic.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.loglogistic.LogLogistic.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.loglogistic.LogLogistic.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.loglogistic.LogLogistic.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.loglogistic.LogLogistic.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.loglogistic.LogLogistic.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.loglogistic.LogLogistic.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.loglogistic\_3p module[](#module-phitter.continuous.continuous_distributions.loglogistic_3p "Link to this heading")

*class* phitter.continuous.continuous\_distributions.loglogistic\_3p.LogLogistic3P(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.loglogistic_3p.LogLogistic3P "Link to this definition")
:   Bases: `object`

## Loglogistic distribution
## - Parameters LogLogistic3P Distribution: {“loc”: \*, “alpha”: \*, “beta”: \*}
## - <https://phitter.io/distributions/continuous/loglogistic_3p>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.loglogistic_3p.LogLogistic3P.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.loglogistic_3p.LogLogistic3P.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.loglogistic_3p.LogLogistic3P.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“loc”: \*, “alpha”: \*, “beta”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.loglogistic_3p.LogLogistic3P.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.loglogistic_3p.LogLogistic3P.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.loglogistic_3p.LogLogistic3P.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.loglogistic_3p.LogLogistic3P.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.loglogistic_3p.LogLogistic3P.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.loglogistic_3p.LogLogistic3P.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.loglogistic_3p.LogLogistic3P.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.loglogistic_3p.LogLogistic3P.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.loglogistic_3p.LogLogistic3P.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.loglogistic_3p.LogLogistic3P.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.loglogistic_3p.LogLogistic3P.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.loglogistic_3p.LogLogistic3P.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.loglogistic_3p.LogLogistic3P.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.loglogistic_3p.LogLogistic3P.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.loglogistic_3p.LogLogistic3P.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.lognormal module[](#module-phitter.continuous.continuous_distributions.lognormal "Link to this heading")

*class* phitter.continuous.continuous\_distributions.lognormal.LogNormal(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.lognormal.LogNormal "Link to this definition")
:   Bases: `object`

## Lognormal distribution
## - Parameters LogNormal Distribution: {“mu”: \*, “sigma”: \*}
## - <https://phitter.io/distributions/continuous/lognormal>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.lognormal.LogNormal.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.lognormal.LogNormal.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.lognormal.LogNormal.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*dict*) – {“mu”: \* , “variance”: \* , “skewness”: \* , “kurtosis”: \* , “data”: \* }

#### Returns
**parameters**

#### Return type
{“mu”: \*, “sigma”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.lognormal.LogNormal.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.lognormal.LogNormal.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.lognormal.LogNormal.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.lognormal.LogNormal.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.lognormal.LogNormal.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.lognormal.LogNormal.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.lognormal.LogNormal.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.lognormal.LogNormal.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.lognormal.LogNormal.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.lognormal.LogNormal.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.lognormal.LogNormal.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.lognormal.LogNormal.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.lognormal.LogNormal.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.lognormal.LogNormal.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.lognormal.LogNormal.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.maxwell module[](#module-phitter.continuous.continuous_distributions.maxwell "Link to this heading")

*class* phitter.continuous.continuous\_distributions.maxwell.Maxwell(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.maxwell.Maxwell "Link to this definition")
:   Bases: `object`

## Maxwell distribution
## - Parameters Maxwell Distribution: {“alpha”: \*, “loc”: \*}
## - <https://phitter.io/distributions/continuous/maxwell>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.maxwell.Maxwell.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.maxwell.Maxwell.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.maxwell.Maxwell.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“alpha”: \*, “loc”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.maxwell.Maxwell.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.maxwell.Maxwell.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.maxwell.Maxwell.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.maxwell.Maxwell.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.maxwell.Maxwell.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.maxwell.Maxwell.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.maxwell.Maxwell.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.maxwell.Maxwell.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.maxwell.Maxwell.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.maxwell.Maxwell.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.maxwell.Maxwell.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.maxwell.Maxwell.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.maxwell.Maxwell.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.maxwell.Maxwell.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.maxwell.Maxwell.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.moyal module[](#module-phitter.continuous.continuous_distributions.moyal "Link to this heading")

*class* phitter.continuous.continuous\_distributions.moyal.Moyal(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.moyal.Moyal "Link to this definition")
:   Bases: `object`

## Moyal distribution
## - Parameters Moyal Distribution: {“mu”: \*, “sigma”: \*}
## - <https://phitter.io/distributions/continuous/moyal>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.moyal.Moyal.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.moyal.Moyal.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.moyal.Moyal.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*dict*) – {“mu”: \* , “variance”: \* , “skewness”: \* , “kurtosis”: \* , “data”: \* }

#### Returns
**parameters**

#### Return type
{“mu”: \*, “sigma”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.moyal.Moyal.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.moyal.Moyal.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.moyal.Moyal.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.moyal.Moyal.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.moyal.Moyal.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.moyal.Moyal.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.moyal.Moyal.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.moyal.Moyal.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.moyal.Moyal.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.moyal.Moyal.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.moyal.Moyal.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.moyal.Moyal.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.moyal.Moyal.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.moyal.Moyal.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.moyal.Moyal.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.nakagami module[](#module-phitter.continuous.continuous_distributions.nakagami "Link to this heading")

*class* phitter.continuous.continuous\_distributions.nakagami.Nakagami(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.nakagami.Nakagami "Link to this definition")
:   Bases: `object`

## Nakagami distribution
## - Parameters Nakagami Distribution: {“m”: \*, “omega”: \*}
## - <https://phitter.io/distributions/continuous/nakagami>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.nakagami.Nakagami.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.nakagami.Nakagami.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.nakagami.Nakagami.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“m”: \*, “omega”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.nakagami.Nakagami.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.nakagami.Nakagami.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.nakagami.Nakagami.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.nakagami.Nakagami.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.nakagami.Nakagami.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.nakagami.Nakagami.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.nakagami.Nakagami.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.nakagami.Nakagami.parameter_restrictions "Link to this definition")
Check parameters restriction

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.nakagami.Nakagami.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.nakagami.Nakagami.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.nakagami.Nakagami.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.nakagami.Nakagami.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.nakagami.Nakagami.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.nakagami.Nakagami.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.nakagami.Nakagami.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.non\_central\_chi\_square module[](#module-phitter.continuous.continuous_distributions.non_central_chi_square "Link to this heading")

*class* phitter.continuous.continuous\_distributions.non\_central\_chi\_square.NonCentralChiSquare(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.non_central_chi_square.NonCentralChiSquare "Link to this definition")
:   Bases: `object`

## Non-Central Chi Square distribution
## - Parameters NonCentralChiSquare Distribution: {“lambda”: \*, “n”: \*}
## - <https://phitter.io/distributions/continuous/non_central_chi_square>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.non_central_chi_square.NonCentralChiSquare.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.non_central_chi_square.NonCentralChiSquare.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.non_central_chi_square.NonCentralChiSquare.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“lambda”: \*, “n”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.non_central_chi_square.NonCentralChiSquare.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.non_central_chi_square.NonCentralChiSquare.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.non_central_chi_square.NonCentralChiSquare.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.non_central_chi_square.NonCentralChiSquare.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.non_central_chi_square.NonCentralChiSquare.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.non_central_chi_square.NonCentralChiSquare.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.non_central_chi_square.NonCentralChiSquare.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.non_central_chi_square.NonCentralChiSquare.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.non_central_chi_square.NonCentralChiSquare.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.non_central_chi_square.NonCentralChiSquare.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.non_central_chi_square.NonCentralChiSquare.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.non_central_chi_square.NonCentralChiSquare.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.non_central_chi_square.NonCentralChiSquare.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.non_central_chi_square.NonCentralChiSquare.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.non_central_chi_square.NonCentralChiSquare.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.non\_central\_f module[](#module-phitter.continuous.continuous_distributions.non_central_f "Link to this heading")

*class* phitter.continuous.continuous\_distributions.non\_central\_f.NonCentralF(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.non_central_f.NonCentralF "Link to this definition")
:   Bases: `object`

## Non-Central F distribution
## - Parameters NonCentralF Distribution: {“lambda”: \*, “n1”: \*, “n2”: \*}
## - <https://phitter.io/distributions/continuous/non_central_f>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.non_central_f.NonCentralF.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.non_central_f.NonCentralF.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.non_central_f.NonCentralF.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“lambda”: \*, “n1”: \*, “n2”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.non_central_f.NonCentralF.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.non_central_f.NonCentralF.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.non_central_f.NonCentralF.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.non_central_f.NonCentralF.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.non_central_f.NonCentralF.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.non_central_f.NonCentralF.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.non_central_f.NonCentralF.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.non_central_f.NonCentralF.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.non_central_f.NonCentralF.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.non_central_f.NonCentralF.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.non_central_f.NonCentralF.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.non_central_f.NonCentralF.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.non_central_f.NonCentralF.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.non_central_f.NonCentralF.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.non_central_f.NonCentralF.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.non\_central\_t\_student module[](#module-phitter.continuous.continuous_distributions.non_central_t_student "Link to this heading")

*class* phitter.continuous.continuous\_distributions.non\_central\_t\_student.NonCentralTStudent(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.non_central_t_student.NonCentralTStudent "Link to this definition")
:   Bases: `object`

## Non-Central T Student distribution
## - Parameters NonCentralTStudent Distribution: {“lambda”: \*, “n”: \*, “loc”: \*, “scale”: \*}
## - <https://phitter.io/distributions/continuous/non_central_t_student> Hand-book on Statistical Distributions (pag.116) … Christian Walck

## cdf(*x*)[](#phitter.continuous.continuous_distributions.non_central_t_student.NonCentralTStudent.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.non_central_t_student.NonCentralTStudent.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.non_central_t_student.NonCentralTStudent.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“lambda”: \*, “n”: \*, “loc”: \*, “scale”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.non_central_t_student.NonCentralTStudent.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.non_central_t_student.NonCentralTStudent.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.non_central_t_student.NonCentralTStudent.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.non_central_t_student.NonCentralTStudent.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.non_central_t_student.NonCentralTStudent.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.non_central_t_student.NonCentralTStudent.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.non_central_t_student.NonCentralTStudent.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.non_central_t_student.NonCentralTStudent.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.non_central_t_student.NonCentralTStudent.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.non_central_t_student.NonCentralTStudent.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.non_central_t_student.NonCentralTStudent.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.non_central_t_student.NonCentralTStudent.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.non_central_t_student.NonCentralTStudent.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.non_central_t_student.NonCentralTStudent.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.non_central_t_student.NonCentralTStudent.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.normal module[](#module-phitter.continuous.continuous_distributions.normal "Link to this heading")

*class* phitter.continuous.continuous\_distributions.normal.Normal(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.normal.Normal "Link to this definition")
:   Bases: `object`

## Normal distribution
## - Parameters Normal Distribution: {“mu”: \*, “sigma”: \*}
## - <https://phitter.io/distributions/continuous/normal>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.normal.Normal.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.normal.Normal.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.normal.Normal.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*dict*) – {“mu”: \* , “variance”: \* , “skewness”: \* , “kurtosis”: \* , “data”: \* }

#### Returns
**parameters**

#### Return type
{“mu”: \*, “sigma”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.normal.Normal.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.normal.Normal.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.normal.Normal.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.normal.Normal.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.normal.Normal.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.normal.Normal.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.normal.Normal.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.normal.Normal.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.normal.Normal.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.normal.Normal.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.normal.Normal.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.normal.Normal.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.normal.Normal.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.normal.Normal.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.normal.Normal.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.pareto\_first\_kind module[](#module-phitter.continuous.continuous_distributions.pareto_first_kind "Link to this heading")

*class* phitter.continuous.continuous\_distributions.pareto\_first\_kind.ParetoFirstKind(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.pareto_first_kind.ParetoFirstKind "Link to this definition")
:   Bases: `object`

## Pareto first kind distribution distribution
## - Parameters ParetoFirstKind Distribution: {“alpha”: \*, “xm”: \*, “loc”: \*}
## - <https://phitter.io/distributions/continuous/pareto_first_kind>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.pareto_first_kind.ParetoFirstKind.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.pareto_first_kind.ParetoFirstKind.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.pareto_first_kind.ParetoFirstKind.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“alpha”: \*, “xm”: \*, “loc”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.pareto_first_kind.ParetoFirstKind.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.pareto_first_kind.ParetoFirstKind.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.pareto_first_kind.ParetoFirstKind.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.pareto_first_kind.ParetoFirstKind.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.pareto_first_kind.ParetoFirstKind.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.pareto_first_kind.ParetoFirstKind.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.pareto_first_kind.ParetoFirstKind.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.pareto_first_kind.ParetoFirstKind.parameter_restrictions "Link to this definition")
Check parameters restriction

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.pareto_first_kind.ParetoFirstKind.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.pareto_first_kind.ParetoFirstKind.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.pareto_first_kind.ParetoFirstKind.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.pareto_first_kind.ParetoFirstKind.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.pareto_first_kind.ParetoFirstKind.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.pareto_first_kind.ParetoFirstKind.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.pareto_first_kind.ParetoFirstKind.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.pareto\_second\_kind module[](#module-phitter.continuous.continuous_distributions.pareto_second_kind "Link to this heading")

*class* phitter.continuous.continuous\_distributions.pareto\_second\_kind.ParetoSecondKind(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.pareto_second_kind.ParetoSecondKind "Link to this definition")
:   Bases: `object`

## Pareto second kind distribution Distribution
## Also known as Lomax Distribution or Pareto Type II distributions
## - <https://phitter.io/distributions/continuous/pareto_second_kind>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.pareto_second_kind.ParetoSecondKind.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.pareto_second_kind.ParetoSecondKind.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.pareto_second_kind.ParetoSecondKind.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“alpha”: \*, “xm”: \*, “loc”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.pareto_second_kind.ParetoSecondKind.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.pareto_second_kind.ParetoSecondKind.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.pareto_second_kind.ParetoSecondKind.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.pareto_second_kind.ParetoSecondKind.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.pareto_second_kind.ParetoSecondKind.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.pareto_second_kind.ParetoSecondKind.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.pareto_second_kind.ParetoSecondKind.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.pareto_second_kind.ParetoSecondKind.parameter_restrictions "Link to this definition")
Check parameters restriction

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.pareto_second_kind.ParetoSecondKind.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.pareto_second_kind.ParetoSecondKind.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.pareto_second_kind.ParetoSecondKind.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.pareto_second_kind.ParetoSecondKind.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.pareto_second_kind.ParetoSecondKind.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.pareto_second_kind.ParetoSecondKind.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.pareto_second_kind.ParetoSecondKind.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.pert module[](#module-phitter.continuous.continuous_distributions.pert "Link to this heading")

*class* phitter.continuous.continuous\_distributions.pert.Pert(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.pert.Pert "Link to this definition")
:   Bases: `object`

## Pert distribution
## - Parameters Pert Distribution: {“a”: \*, “b”: \*, “c”: \*}
## - <https://phitter.io/distributions/continuous/pert>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.pert.Pert.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.pert.Pert.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.pert.Pert.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by solving the equations of the measures expected
- for this distribution.The number of equations to consider is equal to the number
- of parameters.

#### Parameters
**continuous\_measures** (*dict*) – {“mean”: \* , “variance”: \* , “skewness”: \* , “kurtosis”: \* , “median”: \* , “b”: \* }

#### Returns
**parameters**

#### Return type
{“a”: \*, “b”: \*, “c”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.pert.Pert.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.pert.Pert.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.pert.Pert.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.pert.Pert.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.pert.Pert.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.pert.Pert.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.pert.Pert.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.pert.Pert.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.pert.Pert.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.pert.Pert.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.pert.Pert.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.pert.Pert.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.pert.Pert.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.pert.Pert.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.pert.Pert.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.power\_function module[](#module-phitter.continuous.continuous_distributions.power_function "Link to this heading")

*class* phitter.continuous.continuous\_distributions.power\_function.PowerFunction(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.power_function.PowerFunction "Link to this definition")
:   Bases: `object`

## Power function distribution
## - Parameters PowerFunction Distribution: {“alpha”: \*, “a”: \*, “b”: \*}
## - <https://phitter.io/distributions/continuous/power_function>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.power_function.PowerFunction.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.power_function.PowerFunction.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.power_function.PowerFunction.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by solving the equations of the measures expected
- for this distribution.The number of equations to consider is equal to the number
- of parameters.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“alpha”: \*, “a”: \*, “b”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.power_function.PowerFunction.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.power_function.PowerFunction.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.power_function.PowerFunction.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.power_function.PowerFunction.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.power_function.PowerFunction.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.power_function.PowerFunction.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.power_function.PowerFunction.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.power_function.PowerFunction.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.power_function.PowerFunction.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.power_function.PowerFunction.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.power_function.PowerFunction.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.power_function.PowerFunction.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.power_function.PowerFunction.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.power_function.PowerFunction.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.power_function.PowerFunction.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.rayleigh module[](#module-phitter.continuous.continuous_distributions.rayleigh "Link to this heading")

*class* phitter.continuous.continuous\_distributions.rayleigh.Rayleigh(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.rayleigh.Rayleigh "Link to this definition")
:   Bases: `object`

## Rayleigh distribution
## - Parameters Rayleigh Distribution: {“gamma”: \*, “sigma”: \*}
## - <https://phitter.io/distributions/continuous/rayleigh>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.rayleigh.Rayleigh.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.rayleigh.Rayleigh.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.rayleigh.Rayleigh.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by solving the equations of the measures expected
- for this distribution.The number of equations to consider is equal to the number
- of parameters.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“gamma”: \*, “sigma”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.rayleigh.Rayleigh.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.rayleigh.Rayleigh.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.rayleigh.Rayleigh.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.rayleigh.Rayleigh.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.rayleigh.Rayleigh.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.rayleigh.Rayleigh.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.rayleigh.Rayleigh.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.rayleigh.Rayleigh.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.rayleigh.Rayleigh.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.rayleigh.Rayleigh.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.rayleigh.Rayleigh.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.rayleigh.Rayleigh.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.rayleigh.Rayleigh.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.rayleigh.Rayleigh.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.rayleigh.Rayleigh.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.reciprocal module[](#module-phitter.continuous.continuous_distributions.reciprocal "Link to this heading")

*class* phitter.continuous.continuous\_distributions.reciprocal.Reciprocal(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.reciprocal.Reciprocal "Link to this definition")
:   Bases: `object`

## Reciprocal distribution
## - Parameters Reciprocal Distribution: {“a”: \*, “b”: \*}
## - <https://phitter.io/distributions/continuous/reciprocal>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.reciprocal.Reciprocal.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.reciprocal.Reciprocal.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.reciprocal.Reciprocal.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“a”: \*, “b”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.reciprocal.Reciprocal.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.reciprocal.Reciprocal.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.reciprocal.Reciprocal.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.reciprocal.Reciprocal.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.reciprocal.Reciprocal.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.reciprocal.Reciprocal.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.reciprocal.Reciprocal.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.reciprocal.Reciprocal.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.reciprocal.Reciprocal.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.reciprocal.Reciprocal.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.reciprocal.Reciprocal.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.reciprocal.Reciprocal.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.reciprocal.Reciprocal.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.reciprocal.Reciprocal.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.reciprocal.Reciprocal.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.rice module[](#module-phitter.continuous.continuous_distributions.rice "Link to this heading")

*class* phitter.continuous.continuous\_distributions.rice.Rice(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.rice.Rice "Link to this definition")
:   Bases: `object`

## Rice distribution
## - Parameters Rice Distribution: {“v”: \*, “sigma”: \*}
## - <https://phitter.io/distributions/continuous/rice>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.rice.Rice.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.rice.Rice.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.rice.Rice.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by solving the equations of the measures expected
- for this distribution.The number of equations to consider is equal to the number
- of parameters.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“v”: \*, “sigma”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.rice.Rice.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.rice.Rice.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.rice.Rice.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.rice.Rice.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.rice.Rice.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.rice.Rice.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.rice.Rice.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.rice.Rice.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.rice.Rice.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.rice.Rice.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.rice.Rice.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.rice.Rice.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.rice.Rice.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.rice.Rice.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.rice.Rice.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.semicircular module[](#module-phitter.continuous.continuous_distributions.semicircular "Link to this heading")

*class* phitter.continuous.continuous\_distributions.semicircular.Semicircular(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.semicircular.Semicircular "Link to this definition")
:   Bases: `object`

## Semicicrcular Distribution
## - <https://phitter.io/distributions/continuous/semicircular>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.semicircular.Semicircular.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.semicircular.Semicircular.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.semicircular.Semicircular.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*dict*) – {“mu”: \* , “variance”: \* , “skewness”: \* , “kurtosis”: \* , “data”: \* }

#### Returns
**parameters**

#### Return type
{“loc”: \*, “R”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.semicircular.Semicircular.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.semicircular.Semicircular.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.semicircular.Semicircular.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.semicircular.Semicircular.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.semicircular.Semicircular.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.semicircular.Semicircular.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.semicircular.Semicircular.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.semicircular.Semicircular.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.semicircular.Semicircular.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.semicircular.Semicircular.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.semicircular.Semicircular.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.semicircular.Semicircular.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.semicircular.Semicircular.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.semicircular.Semicircular.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.semicircular.Semicircular.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.t\_student module[](#module-phitter.continuous.continuous_distributions.t_student "Link to this heading")

*class* phitter.continuous.continuous\_distributions.t\_student.TStudent(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.t_student.TStudent "Link to this definition")
:   Bases: `object`

## T distribution
## - Parameters TStudent Distribution: {“df”: \*}
## - <https://phitter.io/distributions/continuous/t_student>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.t_student.TStudent.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.t_student.TStudent.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.t_student.TStudent.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by solving the equations of the measures expected
- for this distribution.The number of equations to consider is equal to the number
- of parameters.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“df”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.t_student.TStudent.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.t_student.TStudent.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.t_student.TStudent.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.t_student.TStudent.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.t_student.TStudent.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.t_student.TStudent.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.t_student.TStudent.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.t_student.TStudent.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.t_student.TStudent.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.t_student.TStudent.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.t_student.TStudent.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.t_student.TStudent.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.t_student.TStudent.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.t_student.TStudent.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.t_student.TStudent.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.t\_student\_3p module[](#module-phitter.continuous.continuous_distributions.t_student_3p "Link to this heading")

*class* phitter.continuous.continuous\_distributions.t\_student\_3p.TStudent3P(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.t_student_3p.TStudent3P "Link to this definition")
:   Bases: `object`

## T distribution
## - Parameters TStudent3P Distribution: {“df”: \*, “loc”: \*, “scale”: \*}
## - <https://phitter.io/distributions/continuous/t_student_3p>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.t_student_3p.TStudent3P.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.t_student_3p.TStudent3P.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.t_student_3p.TStudent3P.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by solving the equations of the measures expected
- for this distribution.The number of equations to consider is equal to the number
- of parameters.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“df”: \*, “loc”: \*, “scale”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.t_student_3p.TStudent3P.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.t_student_3p.TStudent3P.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.t_student_3p.TStudent3P.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.t_student_3p.TStudent3P.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.t_student_3p.TStudent3P.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.t_student_3p.TStudent3P.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.t_student_3p.TStudent3P.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.t_student_3p.TStudent3P.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.t_student_3p.TStudent3P.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.t_student_3p.TStudent3P.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.t_student_3p.TStudent3P.ppf "Link to this definition")

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.t_student_3p.TStudent3P.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.t_student_3p.TStudent3P.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.t_student_3p.TStudent3P.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.t_student_3p.TStudent3P.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.trapezoidal module[](#module-phitter.continuous.continuous_distributions.trapezoidal "Link to this heading")

*class* phitter.continuous.continuous\_distributions.trapezoidal.Trapezoidal(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.trapezoidal.Trapezoidal "Link to this definition")
:   Bases: `object`

## Trapezoidal distribution
## - Parameters Trapezoidal Distribution: {“a”: \*, “b”: \*, “c”: \*, “d”: \*}
## - <https://phitter.io/distributions/continuous/trapezoidal>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.trapezoidal.Trapezoidal.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.trapezoidal.Trapezoidal.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.trapezoidal.Trapezoidal.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“a”: \*, “b”: \*, “c”: \*, “d”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.trapezoidal.Trapezoidal.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.trapezoidal.Trapezoidal.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.trapezoidal.Trapezoidal.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.trapezoidal.Trapezoidal.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.trapezoidal.Trapezoidal.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.trapezoidal.Trapezoidal.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.trapezoidal.Trapezoidal.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.trapezoidal.Trapezoidal.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.trapezoidal.Trapezoidal.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.trapezoidal.Trapezoidal.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.trapezoidal.Trapezoidal.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.trapezoidal.Trapezoidal.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.trapezoidal.Trapezoidal.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.trapezoidal.Trapezoidal.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.trapezoidal.Trapezoidal.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.triangular module[](#module-phitter.continuous.continuous_distributions.triangular "Link to this heading")

*class* phitter.continuous.continuous\_distributions.triangular.Triangular(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.triangular.Triangular "Link to this definition")
:   Bases: `object`

## Triangular distribution
## - Parameters Triangular Distribution: {“a”: \*, “b”: \*, “c”: \*}
## - <https://phitter.io/distributions/continuous/triangular>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.triangular.Triangular.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.triangular.Triangular.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.triangular.Triangular.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“a”: \*, “b”: \*, “c”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.triangular.Triangular.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.triangular.Triangular.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.triangular.Triangular.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.triangular.Triangular.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.triangular.Triangular.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.triangular.Triangular.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.triangular.Triangular.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.triangular.Triangular.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.triangular.Triangular.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.triangular.Triangular.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.triangular.Triangular.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.triangular.Triangular.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.triangular.Triangular.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.triangular.Triangular.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.triangular.Triangular.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.uniform module[](#module-phitter.continuous.continuous_distributions.uniform "Link to this heading")

*class* phitter.continuous.continuous\_distributions.uniform.Uniform(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.uniform.Uniform "Link to this definition")
:   Bases: `object`

## Uniform distribution
## - Parameters Uniform Distribution: {“a”: \*, “b”: \*}
## - <https://phitter.io/distributions/continuous/uniform>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.uniform.Uniform.cdf "Link to this definition")
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.uniform.Uniform.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.uniform.Uniform.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“a”: \*, “b”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.uniform.Uniform.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.uniform.Uniform.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.uniform.Uniform.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.uniform.Uniform.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.uniform.Uniform.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.uniform.Uniform.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.uniform.Uniform.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.uniform.Uniform.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.uniform.Uniform.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.uniform.Uniform.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.uniform.Uniform.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.uniform.Uniform.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.uniform.Uniform.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.uniform.Uniform.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.uniform.Uniform.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.weibull module[](#module-phitter.continuous.continuous_distributions.weibull "Link to this heading")

*class* phitter.continuous.continuous\_distributions.weibull.Weibull(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.weibull.Weibull "Link to this definition")
:   Bases: `object`

## Weibull distribution
## - Parameters Weibull Distribution: {“alpha”: \*, “beta”: \*}
## - <https://phitter.io/distributions/continuous/weibull>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.weibull.Weibull.cdf "Link to this definition")
Cumulative distribution function.
- Calculated with known formula.

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.weibull.Weibull.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.weibull.Weibull.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“alpha”: \*, “beta”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.weibull.Weibull.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.weibull.Weibull.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.weibull.Weibull.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.weibull.Weibull.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.weibull.Weibull.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.weibull.Weibull.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.weibull.Weibull.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.weibull.Weibull.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.weibull.Weibull.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.weibull.Weibull.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.weibull.Weibull.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.weibull.Weibull.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.weibull.Weibull.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.weibull.Weibull.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.weibull.Weibull.variance "Link to this definition")
Parametric variance

## phitter.continuous.continuous\_distributions.weibull\_3p module[](#module-phitter.continuous.continuous_distributions.weibull_3p "Link to this heading")

*class* phitter.continuous.continuous\_distributions.weibull\_3p.Weibull3P(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)[](#phitter.continuous.continuous_distributions.weibull_3p.Weibull3P "Link to this definition")
:   Bases: `object`

## Weibull distribution
## - Parameters Weibull3P Distribution: {“alpha”: \*, “loc”: \*, “beta”: \*}
## - <https://phitter.io/distributions/continuous/weibull_3p>

## cdf(*x*)[](#phitter.continuous.continuous_distributions.weibull_3p.Weibull3P.cdf "Link to this definition")
Cumulative distribution function.
- Calculated with known formula.

#### Return type
`float` | `ndarray`

## central\_moments(*k*)[](#phitter.continuous.continuous_distributions.weibull_3p.Weibull3P.central_moments "Link to this definition")
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)[](#phitter.continuous.continuous_distributions.weibull_3p.Weibull3P.get_parameters "Link to this definition")
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“alpha”: \*, “loc”: \*, “beta”: \*}

## *property* kurtosis*: float*[](#phitter.continuous.continuous_distributions.weibull_3p.Weibull3P.kurtosis "Link to this definition")
Parametric kurtosis

## *property* mean*: float*[](#phitter.continuous.continuous_distributions.weibull_3p.Weibull3P.mean "Link to this definition")
Parametric mean

## *property* median*: float*[](#phitter.continuous.continuous_distributions.weibull_3p.Weibull3P.median "Link to this definition")
Parametric median

## *property* mode*: float*[](#phitter.continuous.continuous_distributions.weibull_3p.Weibull3P.mode "Link to this definition")
Parametric mode

## *property* name[](#phitter.continuous.continuous_distributions.weibull_3p.Weibull3P.name "Link to this definition")

## non\_central\_moments(*k*)[](#phitter.continuous.continuous_distributions.weibull_3p.Weibull3P.non_central_moments "Link to this definition")
Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx

#### Return type
`float` | `None`

## *property* num\_parameters*: int*[](#phitter.continuous.continuous_distributions.weibull_3p.Weibull3P.num_parameters "Link to this definition")
Number of parameters of the distribution

## parameter\_restrictions()[](#phitter.continuous.continuous_distributions.weibull_3p.Weibull3P.parameter_restrictions "Link to this definition")
Check parameters restrictions

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*[](#phitter.continuous.continuous_distributions.weibull_3p.Weibull3P.parameters_example "Link to this definition")

## pdf(*x*)[](#phitter.continuous.continuous_distributions.weibull_3p.Weibull3P.pdf "Link to this definition")
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)[](#phitter.continuous.continuous_distributions.weibull_3p.Weibull3P.ppf "Link to this definition")
Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x

#### Return type
`float` | `ndarray`

## sample(*n*, *seed=None*)[](#phitter.continuous.continuous_distributions.weibull_3p.Weibull3P.sample "Link to this definition")
Sample of n elements of ditribution

#### Return type
`ndarray`

## *property* skewness*: float*[](#phitter.continuous.continuous_distributions.weibull_3p.Weibull3P.skewness "Link to this definition")
Parametric skewness

## *property* standard\_deviation*: float*[](#phitter.continuous.continuous_distributions.weibull_3p.Weibull3P.standard_deviation "Link to this definition")
Parametric standard deviation

## *property* variance*: float*[](#phitter.continuous.continuous_distributions.weibull_3p.Weibull3P.variance "Link to this definition")
Parametric variance

## Module contents[](#module-phitter.continuous.continuous_distributions "Link to this heading")

[Previous](phitter.continuous.html "phitter.continuous package")
[Next](phitter.continuous.continuous_measures.html "phitter.continuous.continuous_measures package")

---

© Copyright .

Built with [Sphinx](https://www.sphinx-doc.org/) using a
[theme](https://github.com/readthedocs/sphinx_rtd_theme)
provided by [Read the Docs](https://readthedocs.org).