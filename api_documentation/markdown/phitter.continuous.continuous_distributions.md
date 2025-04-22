* [phitter](modules.html)
* [phitter package](phitter.html)
* [phitter.continuous package](phitter.continuous.html)
* phitter.continuous.continuous\_distributions package
* [View page source](_sources/phitter.continuous.continuous_distributions.rst.txt)

---

# phitter.continuous.continuous\_distributions package

## Submodules

## phitter.continuous.continuous\_distributions.alpha module

*class* phitter.continuous.continuous\_distributions.alpha.Alpha(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Alpha distribution
## - Parameters Alpha Distribution: {“alpha”: \*, “loc”: \*, “scale”: \*}
## - <https://phitter.io/distributions/continuous/alpha>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.arcsine module

*class* phitter.continuous.continuous\_distributions.arcsine.Arcsine(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Arcsine distribution
## - Parameters Arcsine Distribution: {“a”: \*, “b”: \*}
## - <https://phitter.io/distributions/continuous/arcsine>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.argus module

*class* phitter.continuous.continuous\_distributions.argus.Argus(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Argus distribution
## - Parameters Argus Distribution: {“chi”: \*, “loc”: \*, “scale”: \*}
## - <https://phitter.io/distributions/continuous/argus>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.beta module

*class* phitter.continuous.continuous\_distributions.beta.Beta(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Beta distribution
## - Parameters Beta Distribution: {“alpha”: \*, “beta”: \*, “A”: \*, “B”: \*}
## - <https://phitter.io/distributions/continuous/beta>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.beta\_prime module

*class* phitter.continuous.continuous\_distributions.beta\_prime.BetaPrime(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Beta Prime Distribution
## - Parameters BetaPrime Distribution: {“alpha”: \*, “beta”: \*}
## - <https://phitter.io/distributions/continuous/beta_prime>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.beta\_prime\_4p module

*class* phitter.continuous.continuous\_distributions.beta\_prime\_4p.BetaPrime4P(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Beta Prime 4P Distribution
## - Parameters BetaPrime4P Distribution: {“alpha”: \*, “beta”: \*, “loc”: \*, “scale”: \*}
## - <https://phitter.io/distributions/continuous/beta_prime_4p>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.bradford module

*class* phitter.continuous.continuous\_distributions.bradford.Bradford(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Bradford distribution
## - Parameters Bradford Distribution: {“c”: \*, “min”: \*, “max”: \*}
## - <https://phitter.io/distributions/continuous/bradford>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“c”: \*, “min”: \*, “max”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.burr module

*class* phitter.continuous.continuous\_distributions.burr.Burr(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Burr distribution
## - Parameters Burr Distribution: {“A”: \*, “B”: \*, “C”: \*}
## - <https://phitter.io/distributions/continuous/burr>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“A”: \*, “B”: \*, “C”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.burr\_4p module

*class* phitter.continuous.continuous\_distributions.burr\_4p.Burr4P(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Burr distribution
## - Parameters Burr4P Distribution: {“A”: \*, “B”: \*, “C”: \*, “loc”: \*}
## - <https://phitter.io/distributions/continuous/burr_4p>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“A”: \*, “B”: \*, “C”: \*, “loc”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.cauchy module

*class* phitter.continuous.continuous\_distributions.cauchy.Cauchy(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Cauchy distribution
## - Parameters Cauchy Distribution: {“x0”: \*, “gamma”: \*}
## - <https://phitter.io/distributions/continuous/cauchy>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“x0”: \*, “gamma”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.chi\_square module

*class* phitter.continuous.continuous\_distributions.chi\_square.ChiSquare(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Chi Square distribution
## - Parameters ChiSquare Distribution: {“df”: \*}
## - <https://phitter.io/distributions/continuous/chi_square>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“df”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.chi\_square\_3p module

*class* phitter.continuous.continuous\_distributions.chi\_square\_3p.ChiSquare3P(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Chi Square distribution
## - Parameters ChiSquare3P Distribution: {“df”: \*, “loc”: \*, “scale”: \*}
## - <https://phitter.io/distributions/continuous/chi_square_3p>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“df”: \*, “loc”: \*, “scale”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.dagum module

*class* phitter.continuous.continuous\_distributions.dagum.Dagum(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Dagum distribution
## - Parameters Dagum Distribution: {“a”: \*, “b”: \*, “p”: \*}
## - <https://phitter.io/distributions/continuous/dagum>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“a”: \*, “b”: \*, “p”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.dagum\_4p module

*class* phitter.continuous.continuous\_distributions.dagum\_4p.Dagum4P(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Dagum distribution
## - Parameters Dagum4P Distribution: {“a”: \*, “b”: \*, “p”: \*, “loc”: \*}
## - <https://phitter.io/distributions/continuous/dagum_4p>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“a”: \*, “b”: \*, “p”: \*, “loc”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.erlang module

*class* phitter.continuous.continuous\_distributions.erlang.Erlang(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Erlang distribution
## - Parameters Erlang Distribution: {“k”: \*, “beta”: \*}
## - <https://phitter.io/distributions/continuous/erlang>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“k”: \*, “beta”: \*}

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
Check parameters restriction

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.erlang\_3p module

*class* phitter.continuous.continuous\_distributions.erlang\_3p.Erlang3P(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Erlang 3p distribution
## - Parameters Erlang3P Distribution: {“k”: \*, “beta”: \*, “loc”: \*}
## - <https://phitter.io/distributions/continuous/erlang_3p>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“k”: \*, “beta”: \*, “loc”: \*}

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
Check parameters restriction

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.error\_function module

*class* phitter.continuous.continuous\_distributions.error\_function.ErrorFunction(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Error Function distribution
## - Parameters ErrorFunction Distribution: {“h”: \*}
## - <https://phitter.io/distributions/continuous/error_function>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“h”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.exponential module

*class* phitter.continuous.continuous\_distributions.exponential.Exponential(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Exponential distribution
## - Parameters Exponential Distribution: {“lambda”: \*}
## - <https://phitter.io/distributions/continuous/exponential>

## cdf(*x*)
Cumulative distribution function.
- Calculated with known formula.

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.exponential\_2p module

*class* phitter.continuous.continuous\_distributions.exponential\_2p.Exponential2P(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Exponential distribution
## - Parameters Exponential2P Distribution: {“lambda”: \*, “loc”: \*}
## - <https://phitter.io/distributions/continuous/exponential_2p>

## cdf(*x*)
Cumulative distribution function.
- Calculated with known formula.

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.f module

*class* phitter.continuous.continuous\_distributions.f.F(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## F distribution
## - Parameters F Distribution: {“df1”: \*, “df2”: \*}
## - <https://phitter.io/distributions/continuous/f>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“df1”: \*, “df2”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.f\_4p module

*class* phitter.continuous.continuous\_distributions.f\_4p.F4P(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## F distribution
## - Parameters F4P Distribution: {“df1”: \*, “df2”: \*, “loc”: \*, “scale”: \*}
## - <https://phitter.io/distributions/continuous/f_4p>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“df1”: \*, “df2”: \*, “loc”: \*, “scale”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.fatigue\_life module

*class* phitter.continuous.continuous\_distributions.fatigue\_life.FatigueLife(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Fatigue life Distribution
## Also known as Birnbaum-Saunders distribution
## - Parameters FatigueLife Distribution: {“gamma”: \*, “loc”: \*, “scale”: \*}
## - <https://phitter.io/distributions/continuous/fatigue_life>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“gamma”: \*, “loc”: \*, “scale”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.folded\_normal module

*class* phitter.continuous.continuous\_distributions.folded\_normal.FoldedNormal(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Folded Normal Distribution
## - <https://phitter.io/distributions/continuous/folded_normal>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*dict*) – {“mu”: \* , “variance”: \* , “skewness”: \* , “kurtosis”: \* , “data”: \* }

#### Returns
**parameters**

#### Return type
{“mu”: \*, “sigma”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.frechet module

*class* phitter.continuous.continuous\_distributions.frechet.Frechet(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Fréchet Distribution
## Also known as inverse Weibull distribution (Scipy name)
## - <https://phitter.io/distributions/continuous/frechet>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“alpha”: \*, “loc”: \*, “scale”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.gamma module

*class* phitter.continuous.continuous\_distributions.gamma.Gamma(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Gamma distribution
## - Parameters Gamma Distribution: {“alpha”: \*, “beta”: \*}
## - <https://phitter.io/distributions/continuous/gamma>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“alpha”: \*, “beta”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.gamma\_3p module

*class* phitter.continuous.continuous\_distributions.gamma\_3p.Gamma3P(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Gamma distribution
## - Parameters Gamma3P Distribution: {“alpha”: \*, “loc”: \*, “beta”: \*}
## - <https://phitter.io/distributions/continuous/gamma_3p>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“alpha”: \*, “loc”: \*, “beta”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.generalized\_extreme\_value module

*class* phitter.continuous.continuous\_distributions.generalized\_extreme\_value.GeneralizedExtremeValue(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Generalized Extreme Value Distribution
## - <https://phitter.io/distributions/continuous/generalized_extreme_value>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“xi”: \*, “mu”: \*, “sigma”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.generalized\_gamma module

*class* phitter.continuous.continuous\_distributions.generalized\_gamma.GeneralizedGamma(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Generalized Gamma Distribution
## - <https://phitter.io/distributions/continuous/generalized_gamma>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“a”: \*, “d”: \*, “p”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.generalized\_gamma\_4p module

*class* phitter.continuous.continuous\_distributions.generalized\_gamma\_4p.GeneralizedGamma4P(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Generalized Gamma Distribution
## - <https://phitter.io/distributions/continuous/generalized_gamma_4p>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“a”: \*, “d”: \*, “p”: \*, “loc”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.generalized\_logistic module

*class* phitter.continuous.continuous\_distributions.generalized\_logistic.GeneralizedLogistic(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Generalized Logistic Distribution

## * <https://phitter.io/distributions/continuous/generalized_logistic>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“loc”: \*, “scale”: \*, “c”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.generalized\_normal module

*class* phitter.continuous.continuous\_distributions.generalized\_normal.GeneralizedNormal(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Generalized normal distribution
## - Parameters GeneralizedNormal Distribution: {“beta”: \*, “mu”: \*, “alpha”: \*}
## - <https://phitter.io/distributions/continuous/generalized_normal> This distribution is known whit the following names:
## \* Error Distribution
## \* Exponential Power Distribution
## \* Generalized Error Distribution (GED)
## \* Generalized Gaussian distribution (GGD)
## \* Subbotin distribution

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“beta”: \*, “mu”: \*, “alpha”: \*}

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
Check parameters restriction

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.generalized\_pareto module

*class* phitter.continuous.continuous\_distributions.generalized\_pareto.GeneralizedPareto(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Generalized Pareto distribution
## - Parameters GeneralizedPareto Distribution: {“c”: \*, “mu”: \*, “sigma”: \*}
## - <https://phitter.io/distributions/continuous/generalized_pareto>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“c”: \*, “mu”: \*, “sigma”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.gibrat module

*class* phitter.continuous.continuous\_distributions.gibrat.Gibrat(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Gibrat distribution
## - Parameters Gibrat Distribution: {“loc”: \*, “scale”: \*}
## - <https://phitter.io/distributions/continuous/gibrat>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.gumbel\_left module

*class* phitter.continuous.continuous\_distributions.gumbel\_left.GumbelLeft(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Gumbel Left Distribution
## Gumbel Min Distribution
## Extreme Value Minimum Distribution
## - <https://phitter.io/distributions/continuous/gumbel_left>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“mu”: \*, “sigma”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.gumbel\_right module

*class* phitter.continuous.continuous\_distributions.gumbel\_right.GumbelRight(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Gumbel Right Distribution
## Gumbel Max Distribution
## Extreme Value Maximum Distribution

## * <https://phitter.io/distributions/continuous/gumbel_right>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“mu”: \*, “sigma”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.half\_normal module

*class* phitter.continuous.continuous\_distributions.half\_normal.HalfNormal(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Half Normal Distribution
## - <https://phitter.io/distributions/continuous/half_normal>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*dict*) – {“mu”: \* , “variance”: \* , “skewness”: \* , “kurtosis”: \* , “data”: \* }

#### Returns
**parameters**

#### Return type
{“mu”: \*, “sigma”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.hyperbolic\_secant module

*class* phitter.continuous.continuous\_distributions.hyperbolic\_secant.HyperbolicSecant(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Hyperbolic Secant distribution
## - Parameters HyperbolicSecant Distribution: {“mu”: \*, “sigma”: \*}
## - <https://phitter.io/distributions/continuous/hyperbolic_secant>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*dict*) – {“mu”: \* , “variance”: \* , “skewness”: \* , “kurtosis”: \* , “data”: \* }

#### Returns
**parameters**

#### Return type
{“mu”: \*, “sigma”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.inverse\_gamma module

*class* phitter.continuous.continuous\_distributions.inverse\_gamma.InverseGamma(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Inverse Gamma Distribution
## Also known Pearson Type 5 distribution
## - Parameters InverseGamma Distribution: {“alpha”: \*, “beta”: \*}
## - <https://phitter.io/distributions/continuous/inverse_gamma>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“alpha”: \*, “beta”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.inverse\_gamma\_3p module

*class* phitter.continuous.continuous\_distributions.inverse\_gamma\_3p.InverseGamma3P(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Inverse Gamma Distribution
## Also known Pearson Type 5 distribution
## - Parameters InverseGamma3P Distribution: {“alpha”: \*, “beta”: \*, “loc”: \*}
## - <https://phitter.io/distributions/continuous/inverse_gamma_3p>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“alpha”: \*, “beta”: \*, “loc”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.inverse\_gaussian module

*class* phitter.continuous.continuous\_distributions.inverse\_gaussian.InverseGaussian(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Inverse Gaussian Distribution
## Also known like Wald distribution
## - Parameters InverseGaussian Distribution: {“mu”: \*, “lambda”: \*}
## - <https://phitter.io/distributions/continuous/inverse_gaussian>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*dict*) – {“mu”: \* , “variance”: \* , “skewness”: \* , “kurtosis”: \* , “data”: \* }

#### Returns
**parameters**

#### Return type
{“mu”: \*, “lambda”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.inverse\_gaussian\_3p module

*class* phitter.continuous.continuous\_distributions.inverse\_gaussian\_3p.InverseGaussian3P(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Inverse Gaussian Distribution
## Also known like Wald distribution
## - Parameters InverseGaussian3P Distribution: {“mu”: \*, “lambda”: \*, “loc”: \*}
## - <https://phitter.io/distributions/continuous/inverse_gaussian_3p>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*dict*) – {“mu”: \* , “variance”: \* , “skewness”: \* , “kurtosis”: \* , “data”: \* }

#### Returns
**parameters**

#### Return type
{“mu”: \*, “lambda”: \*, “loc”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.johnson\_sb module

*class* phitter.continuous.continuous\_distributions.johnson\_sb.JohnsonSB(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Johnson SB distribution
## - Parameters JohnsonSB Distribution: {“xi”: \*, “lambda”: \*, “gamma”: \*, “delta”: \*}
## - <https://phitter.io/distributions/continuous/johnson_sb>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.johnson\_su module

*class* phitter.continuous.continuous\_distributions.johnson\_su.JohnsonSU(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Johnson SU distribution
## - Parameters JohnsonSU Distribution: {“xi”: \*, “lambda”: \*, “gamma”: \*, “delta”: \*}
## - <https://phitter.io/distributions/continuous/johnson_su>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Return type
`dict`[`str`, `float` | `int`]

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.kumaraswamy module

*class* phitter.continuous.continuous\_distributions.kumaraswamy.Kumaraswamy(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Kumaraswami distribution
## - Parameters Kumaraswamy Distribution: {“alpha”: \*, “beta”: \*, “min”: \*, “max”: \*}
## - <https://phitter.io/distributions/continuous/kumaraswamy>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.laplace module

*class* phitter.continuous.continuous\_distributions.laplace.Laplace(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Laplace distribution
## - Parameters Laplace Distribution: {“mu”: \*, “b”: \*}
## - <https://phitter.io/distributions/continuous/laplace>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“mu”: \*, “b”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.levy module

*class* phitter.continuous.continuous\_distributions.levy.Levy(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Levy distribution
## - Parameters Levy Distribution: {“mu”: \*, “c”: \*}
## - <https://phitter.io/distributions/continuous/levy>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“mu”: \*, “c”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.loggamma module

*class* phitter.continuous.continuous\_distributions.loggamma.LogGamma(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## LogGamma distribution
## - Parameters LogGamma Distribution: {“c”: \*, “mu”: \*, “sigma”: \*}
## - <https://phitter.io/distributions/continuous/loggamma>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“c”: \*, “mu”: \*, “sigma”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.logistic module

*class* phitter.continuous.continuous\_distributions.logistic.Logistic(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Logistic distribution
## - Parameters Logistic Distribution: {“mu”: \*, “sigma”: \*}
## - <https://phitter.io/distributions/continuous/logistic>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“mu”: \*, “sigma”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.loglogistic module

*class* phitter.continuous.continuous\_distributions.loglogistic.LogLogistic(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Loglogistic distribution
## - Parameters LogLogistic Distribution: {“alpha”: \*, “beta”: \*}
## - <https://phitter.io/distributions/continuous/loglogistic>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“alpha”: \*, “beta”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.loglogistic\_3p module

*class* phitter.continuous.continuous\_distributions.loglogistic\_3p.LogLogistic3P(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Loglogistic distribution
## - Parameters LogLogistic3P Distribution: {“loc”: \*, “alpha”: \*, “beta”: \*}
## - <https://phitter.io/distributions/continuous/loglogistic_3p>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“loc”: \*, “alpha”: \*, “beta”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.lognormal module

*class* phitter.continuous.continuous\_distributions.lognormal.LogNormal(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Lognormal distribution
## - Parameters LogNormal Distribution: {“mu”: \*, “sigma”: \*}
## - <https://phitter.io/distributions/continuous/lognormal>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*dict*) – {“mu”: \* , “variance”: \* , “skewness”: \* , “kurtosis”: \* , “data”: \* }

#### Returns
**parameters**

#### Return type
{“mu”: \*, “sigma”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.maxwell module

*class* phitter.continuous.continuous\_distributions.maxwell.Maxwell(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Maxwell distribution
## - Parameters Maxwell Distribution: {“alpha”: \*, “loc”: \*}
## - <https://phitter.io/distributions/continuous/maxwell>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“alpha”: \*, “loc”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.moyal module

*class* phitter.continuous.continuous\_distributions.moyal.Moyal(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Moyal distribution
## - Parameters Moyal Distribution: {“mu”: \*, “sigma”: \*}
## - <https://phitter.io/distributions/continuous/moyal>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*dict*) – {“mu”: \* , “variance”: \* , “skewness”: \* , “kurtosis”: \* , “data”: \* }

#### Returns
**parameters**

#### Return type
{“mu”: \*, “sigma”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.nakagami module

*class* phitter.continuous.continuous\_distributions.nakagami.Nakagami(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Nakagami distribution
## - Parameters Nakagami Distribution: {“m”: \*, “omega”: \*}
## - <https://phitter.io/distributions/continuous/nakagami>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“m”: \*, “omega”: \*}

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
Check parameters restriction

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.non\_central\_chi\_square module

*class* phitter.continuous.continuous\_distributions.non\_central\_chi\_square.NonCentralChiSquare(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Non-Central Chi Square distribution
## - Parameters NonCentralChiSquare Distribution: {“lambda”: \*, “n”: \*}
## - <https://phitter.io/distributions/continuous/non_central_chi_square>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“lambda”: \*, “n”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.non\_central\_f module

*class* phitter.continuous.continuous\_distributions.non\_central\_f.NonCentralF(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Non-Central F distribution
## - Parameters NonCentralF Distribution: {“lambda”: \*, “n1”: \*, “n2”: \*}
## - <https://phitter.io/distributions/continuous/non_central_f>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“lambda”: \*, “n1”: \*, “n2”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.non\_central\_t\_student module

*class* phitter.continuous.continuous\_distributions.non\_central\_t\_student.NonCentralTStudent(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Non-Central T Student distribution
## - Parameters NonCentralTStudent Distribution: {“lambda”: \*, “n”: \*, “loc”: \*, “scale”: \*}
## - <https://phitter.io/distributions/continuous/non_central_t_student> Hand-book on Statistical Distributions (pag.116) … Christian Walck

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“lambda”: \*, “n”: \*, “loc”: \*, “scale”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.normal module

*class* phitter.continuous.continuous\_distributions.normal.Normal(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Normal distribution
## - Parameters Normal Distribution: {“mu”: \*, “sigma”: \*}
## - <https://phitter.io/distributions/continuous/normal>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*dict*) – {“mu”: \* , “variance”: \* , “skewness”: \* , “kurtosis”: \* , “data”: \* }

#### Returns
**parameters**

#### Return type
{“mu”: \*, “sigma”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.pareto\_first\_kind module

*class* phitter.continuous.continuous\_distributions.pareto\_first\_kind.ParetoFirstKind(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Pareto first kind distribution distribution
## - Parameters ParetoFirstKind Distribution: {“alpha”: \*, “xm”: \*, “loc”: \*}
## - <https://phitter.io/distributions/continuous/pareto_first_kind>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“alpha”: \*, “xm”: \*, “loc”: \*}

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
Check parameters restriction

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.pareto\_second\_kind module

*class* phitter.continuous.continuous\_distributions.pareto\_second\_kind.ParetoSecondKind(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Pareto second kind distribution Distribution
## Also known as Lomax Distribution or Pareto Type II distributions
## - <https://phitter.io/distributions/continuous/pareto_second_kind>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“alpha”: \*, “xm”: \*, “loc”: \*}

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
Check parameters restriction

#### Return type
`bool`

## *property* parameters\_example*: dict[str, int | float]*

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.pert module

*class* phitter.continuous.continuous\_distributions.pert.Pert(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Pert distribution
## - Parameters Pert Distribution: {“a”: \*, “b”: \*, “c”: \*}
## - <https://phitter.io/distributions/continuous/pert>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.power\_function module

*class* phitter.continuous.continuous\_distributions.power\_function.PowerFunction(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Power function distribution
## - Parameters PowerFunction Distribution: {“alpha”: \*, “a”: \*, “b”: \*}
## - <https://phitter.io/distributions/continuous/power_function>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.rayleigh module

*class* phitter.continuous.continuous\_distributions.rayleigh.Rayleigh(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Rayleigh distribution
## - Parameters Rayleigh Distribution: {“gamma”: \*, “sigma”: \*}
## - <https://phitter.io/distributions/continuous/rayleigh>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.reciprocal module

*class* phitter.continuous.continuous\_distributions.reciprocal.Reciprocal(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Reciprocal distribution
## - Parameters Reciprocal Distribution: {“a”: \*, “b”: \*}
## - <https://phitter.io/distributions/continuous/reciprocal>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.rice module

*class* phitter.continuous.continuous\_distributions.rice.Rice(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Rice distribution
## - Parameters Rice Distribution: {“v”: \*, “sigma”: \*}
## - <https://phitter.io/distributions/continuous/rice>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.semicircular module

*class* phitter.continuous.continuous\_distributions.semicircular.Semicircular(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Semicicrcular Distribution
## - <https://phitter.io/distributions/continuous/semicircular>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*dict*) – {“mu”: \* , “variance”: \* , “skewness”: \* , “kurtosis”: \* , “data”: \* }

#### Returns
**parameters**

#### Return type
{“loc”: \*, “R”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.t\_student module

*class* phitter.continuous.continuous\_distributions.t\_student.TStudent(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## T distribution
## - Parameters TStudent Distribution: {“df”: \*}
## - <https://phitter.io/distributions/continuous/t_student>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.t\_student\_3p module

*class* phitter.continuous.continuous\_distributions.t\_student\_3p.TStudent3P(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## T distribution
## - Parameters TStudent3P Distribution: {“df”: \*, “loc”: \*, “scale”: \*}
## - <https://phitter.io/distributions/continuous/t_student_3p>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
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

## pdf(*x*)
Probability density function

#### Return type
`float` | `ndarray`

## ppf(*u*)

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

## phitter.continuous.continuous\_distributions.trapezoidal module

*class* phitter.continuous.continuous\_distributions.trapezoidal.Trapezoidal(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Trapezoidal distribution
## - Parameters Trapezoidal Distribution: {“a”: \*, “b”: \*, “c”: \*, “d”: \*}
## - <https://phitter.io/distributions/continuous/trapezoidal>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“a”: \*, “b”: \*, “c”: \*, “d”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.triangular module

*class* phitter.continuous.continuous\_distributions.triangular.Triangular(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Triangular distribution
## - Parameters Triangular Distribution: {“a”: \*, “b”: \*, “c”: \*}
## - <https://phitter.io/distributions/continuous/triangular>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“a”: \*, “b”: \*, “c”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.uniform module

*class* phitter.continuous.continuous\_distributions.uniform.Uniform(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Uniform distribution
## - Parameters Uniform Distribution: {“a”: \*, “b”: \*}
## - <https://phitter.io/distributions/continuous/uniform>

## cdf(*x*)
Cumulative distribution function

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.weibull module

*class* phitter.continuous.continuous\_distributions.weibull.Weibull(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Weibull distribution
## - Parameters Weibull Distribution: {“alpha”: \*, “beta”: \*}
## - <https://phitter.io/distributions/continuous/weibull>

## cdf(*x*)
Cumulative distribution function.
- Calculated with known formula.

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“alpha”: \*, “beta”: \*}

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

## pdf(*x*)
Probability density function

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

## phitter.continuous.continuous\_distributions.weibull\_3p module

*class* phitter.continuous.continuous\_distributions.weibull\_3p.Weibull3P(*parameters=None*, *continuous\_measures=None*, *init\_parameters\_examples=False*)
:   Bases: `object`

## Weibull distribution
## - Parameters Weibull3P Distribution: {“alpha”: \*, “loc”: \*, “beta”: \*}
## - <https://phitter.io/distributions/continuous/weibull_3p>

## cdf(*x*)
Cumulative distribution function.
- Calculated with known formula.

#### Return type
`float` | `ndarray`

## central\_moments(*k*)
Parametric central moments. µ’[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx

#### Return type
`float` | `None`

## get\_parameters(*continuous\_measures*)
Calculate proper parameters of the distribution from sample continuous\_measures.
- The parameters are calculated by formula.

#### Parameters
**continuous\_measures** (*MEASUREMESTS*) – attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num\_bins, data

#### Returns
**parameters**

#### Return type
{“alpha”: \*, “loc”: \*, “beta”: \*}

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

## pdf(*x*)
Probability density function

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

[Previous](phitter.continuous.html "phitter.continuous package")
[Next](phitter.continuous.continuous_measures.html "phitter.continuous.continuous_measures package")

---

© Copyright .

Built with [Sphinx](https://www.sphinx-doc.org/) using a
[theme](https://github.com/readthedocs/sphinx_rtd_theme)
provided by [Read the Docs](https://readthedocs.org).