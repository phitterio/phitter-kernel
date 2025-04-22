* [phitter](modules.html)
* [phitter package](phitter.html)
* [phitter.continuous package](phitter.continuous.html)
* phitter.continuous.continuous\_statistical\_tests package
* [View page source](_sources/phitter.continuous.continuous_statistical_tests.rst.txt)

---

# phitter.continuous.continuous\_statistical\_tests package

## Submodules

## phitter.continuous.continuous\_statistical\_tests.continuous\_test\_anderson\_darling module

phitter.continuous.continuous\_statistical\_tests.continuous\_test\_anderson\_darling.evaluate\_continuous\_test\_anderson\_darling(*distribution*, *continuous\_measures*)
:   Anderson Darling test to evaluate that a sample is distributed according to a probability
## distribution.

## The hypothesis that the sample is distributed following the probability distribution
## is not rejected if the test statistic is less than the critical value or equivalently
## if the p-value is less than 0.05

## Parameters
* **data** (*iterable*) – data set
- * **distribution** (*class*) – distribution class initialized whit parameters of distribution and methods
- cdf() and num\_parameters()

## Returns
**result\_test\_ks** –

- 1. test\_statistic(float):
sum over all data(Y) of the value ((2k - 1) / N) \* (ln[Fn(Y[k])] + ln[1 - Fn(Y[N - k + 1])]).
- 2. critical\_value(float):
calculation of the Anderson Darling critical value using Marsaglia - Marsaglia function.
- whit size of sample N as parameter.
- 3. p-value[0,1]:
probability of the test statistic for the Anderson - Darling distribution
- whit size of sample N as parameter.
- 4. rejected(bool):
decision if the null hypothesis is rejected. If it is false, it can be
- considered that the sample is distributed according to the probability
- distribution. If it’s true, no.

## Return type
dict

## References

## [1]

## Marsaglia, G., & Marsaglia, J. (2004).
## Evaluating the anderson - darling distribution.
## Journal of Statistical Software, 9(2), 1 - 5.


## [2]

## Sinclair, C. D., & Spurr, B. D. (1988).
## Approximations to the distribution function of the anderson—darling test statistic.
## Journal of the American Statistical Association, 83(404), 1190 - 1191.


## [3]

## Lewis, P. A. (1961).
## Distribution of the Anderson - Darling statistic.
## The Annals of Mathematical Statistics, 1118 - 1124.

## phitter.continuous.continuous\_statistical\_tests.continuous\_test\_chi\_square module

phitter.continuous.continuous\_statistical\_tests.continuous\_test\_chi\_square.evaluate\_continuous\_test\_chi\_square(*distribution*, *continuous\_measures*)
:   Chi Square test to evaluate that a sample is distributed according to a probability
## distribution.

## The hypothesis that the sample is distributed following the probability distribution
## is not rejected if the test statistic is less than the critical value or equivalently
## if the p-value is less than 0.05

## Parameters
* **data** (*iterable*) – data set
- * **distribution** (*class*) – distribution class initialized whit parameters of distribution and methods
- cdf() and num\_parameters()

## Returns
**result\_test\_chi2** –

- 1. test\_statistic(float):
sum over all classes of the value (expected - observed) ^ 2 / expected
- 2. critical\_value(float):
inverse of the distribution chi square to 0.95 with freedom degrees
- n - 1 minus the number of parameters of the distribution.
- 3. p-value([0,1]):
right - tailed probability of the test statistic for the chi - square distribution
- with the same degrees of freedom as for the critical value calculation.
- 4. rejected(bool):
decision if the null hypothesis is rejected. If it is false, it can be
- considered that the sample is distributed according to the probability
- distribution. If it’s true, no.

## Return type
dict

## phitter.continuous.continuous\_statistical\_tests.continuous\_test\_kolmogorov\_smirnov module

phitter.continuous.continuous\_statistical\_tests.continuous\_test\_kolmogorov\_smirnov.evaluate\_continuous\_test\_kolmogorov\_smirnov(*distribution*, *continuous\_measures*)
:   Kolmogorov Smirnov test to evaluate that a sample is distributed according to a probability
## distribution.

## The hypothesis that the sample is distributed following the probability distribution
## is not rejected if the test statistic is less than the critical value or equivalently
## if the p-value is less than 0.05

## Parameters
* **data** (*iterable*) – data set
- * **distribution** (*class*) – distribution class initialized whit parameters of distribution and methods
- cdf() and num\_parameters()

## Returns
**result\_test\_ks** –

- 1. test\_statistic(float):
sum over all data of the value [|Sn - Fn|](#id4)
- 2. critical\_value(float):
inverse of the kolmogorov - smirnov distribution to 0.95 whit size of
- sample N as parameter.
- 3. p-value[0,1]:
probability of the test statistic for the kolmogorov - smirnov distribution
- whit size of sample N as parameter.
- 4. rejected(bool):
decision if the null hypothesis is rejected. If it is false, it can be
- considered that the sample is distributed according to the probability
- distribution. If it’s true, no.

## Return type
dict

## Module contents

[Previous](phitter.continuous.continuous_measures.html "phitter.continuous.continuous_measures package")
[Next](phitter.discrete.html "phitter.discrete package")

---

© Copyright .

Built with [Sphinx](https://www.sphinx-doc.org/) using a
[theme](https://github.com/readthedocs/sphinx_rtd_theme)
provided by [Read the Docs](https://readthedocs.org).