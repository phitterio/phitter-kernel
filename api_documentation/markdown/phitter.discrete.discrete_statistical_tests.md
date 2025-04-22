* [phitter](modules.html)
* [phitter package](phitter.html)
* [phitter.discrete package](phitter.discrete.html)
* phitter.discrete.discrete\_statistical\_tests package
* [View page source](_sources/phitter.discrete.discrete_statistical_tests.rst.txt)

---

# phitter.discrete.discrete\_statistical\_tests package

## Submodules

## phitter.discrete.discrete\_statistical\_tests.discrete\_test\_chi\_square module

phitter.discrete.discrete\_statistical\_tests.discrete\_test\_chi\_square.evaluate\_discrete\_test\_chi\_square(*distribution*, *discrete\_measures*)
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

## phitter.discrete.discrete\_statistical\_tests.discrete\_test\_kolmogorov\_smirnov module

phitter.discrete.discrete\_statistical\_tests.discrete\_test\_kolmogorov\_smirnov.evaluate\_discrete\_test\_kolmogorov\_smirnov(*distribution*, *discrete\_measures*)
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
sum over all data of the value [|Sn - Fn|](#id1)
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

[Previous](phitter.discrete.discrete_measures.html "phitter.discrete.discrete_measures package")
[Next](phitter.simulation.html "phitter.simulation package")

---

© Copyright .

Built with [Sphinx](https://www.sphinx-doc.org/) using a
[theme](https://github.com/readthedocs/sphinx_rtd_theme)
provided by [Read the Docs](https://readthedocs.org).