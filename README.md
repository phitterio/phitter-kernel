
<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://gist.githubusercontent.com/phitterio/66bc7f3674eac01ae646e30ba697a6d7/raw/e96dbba0eb26b20d35e608fefc3984bd87f0010b/DarkPhitterLogo.svg" width="350">
        <source media="(prefers-color-scheme: light)" srcset="https://gist.githubusercontent.com/phitterio/170ce460d7e766545265772525edecf6/raw/71b4867c6e5683455cf1d68bea5bea7eda55ce7d/LightPhitterLogo.svg" width="350">
        <img alt="phitter-dark-logo" src="https://gist.githubusercontent.com/phitterio/170ce460d7e766545265772525edecf6/raw/71b4867c6e5683455cf1d68bea5bea7eda55ce7d/LightPhitterLogo.svg" width="350">
    </picture>
</p>

<p align="center">
<a href="https://pypi.org/project/phitter" target="_blank">
    <img src="https://img.shields.io/pypi/dm/phitter.svg" alt="Supported Python versions">
</a>
<a href="https://pypi.org/project/phitter" target="_blank">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="Supported Python versions">
</a>
<a href="https://pypi.org/project/phitter" target="_blank">
    <img src="https://img.shields.io/pypi/pyversions/phitter" alt="Supported Python versions">
</a>
</p>



<p>
    Phitter analyzes datasets and determines the best analytical probability distributions that represent them. The Phitter kernel studies over 80 probability distributions, both continuous and discrete, 3 goodness-of-fit tests, and interactive visualizations. For each selected probability distribution, a standard modeling guide is provided along with spreadsheets that detail the methodology for using the chosen distribution in data science, operations research, and artificial intelligence.
</p>
<p>
    This repository contains the implementation of the python library and the kernel of <a href="https://phitter.io">Phitter Web</a>
</p>




## Installation

### Requirements
```console
python: >=3.9
```
### PyPI
```console
pip install phitter
```

## Usage

<a target="_blank" href="https://colab.research.google.com/github/phitterio/phitter-kernel/blob/main/tests/distributions/continuous_distributions.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


### General
```python
import phitter

data: list[int | float] = [...]

phitter_cont = phitter.PHITTER(data)
phitter_cont.fit()
```
### Full continuous  implementation
```python
import phitter

data: list[int | float] = [...]

phitter_cont = phitter.PHITTER(
    data=data,
    fit_type="continuous",
    num_bins=15,
    confidence_level=0.95,
    minimum_sse=1e-2,
    distributions_to_fit=["beta", "normal", "fatigue_life", "triangular"],
)
phitter_cont.fit(n_workers=6)
```
### Full discrete implementation
```python
import phitter

data: list[int | float] = [...]

phitter_disc = phitter.PHITTER(
    data=data,
    fit_type="discrete",
    confidence_level=0.95,
    minimum_sse=1e-2,
    distributions_to_fit=["binomial", "geometric"],
)
phitter_disc.fit(n_workers=2)
```

### Phitter: properties and methods
```python
import phitter
data: list[int | float] = [...]
phitter_cont = phitter.PHITTER(data)
phitter_cont.fit()

phitter_cont.best_distribution -> dict
phitter_cont.sorted_distributions_sse -> dict
phitter_cont.not_rejected_distributions -> dict
phitter_cont.df_sorted_distributions_sse -> pandas.DataFrame
phitter_cont.df_not_rejected_distributions -> pandas.DataFrame
```


### Histogram Plot
```python
import phitter
data: list[int | float] = [...]
phitter_cont = phitter.PHITTER(data)
phitter_cont.fit()

phitter_cont.plot_histogram()
```
<img alt="phitter_histogram" src="https://github.com/phitterio/phitter-kernel/blob/main/utilities/multimedia/histogram.png?raw=true" width="500" />

### Histogram PDF Dsitributions Plot
```python
import phitter
data: list[int | float] = [...]
phitter_cont = phitter.PHITTER(data)
phitter_cont.fit()

phitter_cont.plot_histogram_distributions()
```
<img alt="phitter_histogram" src="https://github.com/phitterio/phitter-kernel/blob/main/utilities/multimedia/histogram_pdf_distributions.png?raw=true" width="500" />

### Histogram PDF Dsitribution Plot
```python
import phitter
data: list[int | float] = [...]
phitter_cont = phitter.PHITTER(data)
phitter_cont.fit()

phitter_cont.plot_distribution("beta")
```
<img alt="phitter_histogram" src="https://github.com/phitterio/phitter-kernel/blob/main/utilities/multimedia/histogram_pdf_distribution.png?raw=true" width="500" />

### ECDF Plot
```python
import phitter
data: list[int | float] = [...]
phitter_cont = phitter.PHITTER(data)
phitter_cont.fit()

phitter_cont.plot_ecdf()
```
<img alt="phitter_histogram" src="https://github.com/phitterio/phitter-kernel/blob/main/utilities/multimedia/ecdf.png?raw=true" width="500" />

### ECDF Distribution Plot
```python
import phitter
data: list[int | float] = [...]
phitter_cont = phitter.PHITTER(data)
phitter_cont.fit()

phitter_cont.plot_ecdf_distribution("beta")
```
<img alt="phitter_histogram" src="https://github.com/phitterio/phitter-kernel/blob/main/utilities/multimedia/ecdf_distribution.png?raw=true" width="500" />


### QQ Plot
```python
import phitter
data: list[int | float] = [...]
phitter_cont = phitter.PHITTER(data)
phitter_cont.fit()

phitter_cont.qq_plot("beta")
```
<img alt="phitter_histogram" src="https://github.com/phitterio/phitter-kernel/blob/main/utilities/multimedia/qq_plot_distribution.png?raw=true" width="500" />

### QQ - Regression Plot 
```python
import phitter
data: list[int | float] = [...]
phitter_cont = phitter.PHITTER(data)
phitter_cont.fit()

phitter_cont.qq_plot_regression("beta")
```
<img alt="phitter_histogram" src="https://github.com/phitterio/phitter-kernel/blob/main/utilities/multimedia/qq_plot_distribution_regression.png?raw=true" width="500" />


### Distributions: Methods and properties 
```python
import phitter

distribution = phitter.continuous.BETA(parameters={"alpha": 5, "beta": 3, "A": 200, "B": 1000})

## CDF, PDF, PPF, PMF receive float or numpy.ndarray. For discrete distributions PMF instead of PDF. Parameters notation are in description of ditribution
distribution.cdf(752) # -> 0.6242831129533498
distribution.pdf(388) # -> 0.0002342575686629883
distribution.ppf(0.623) # -> 751.5512889417921
distribution.sample(2) # -> [550.800114   514.85410326]

## STATS
distribution.mean # -> 700.0
distribution.variance # -> 16666.666666666668
distribution.standard_deviation # -> 129.09944487358058
distribution.skewness # -> -0.3098386676965934
distribution.kurtosis # -> 2.5854545454545454
distribution.median # -> 708.707130841534
distribution.mode # -> 733.3333333333333
```

## Continuous Distributions
#### [1. PDF File Documentation Continuous Distributions](https://github.com/phitterio/phitter-kernel/blob/main/distributions_documentation/continuous/document_continuous_distributions/phitter_continuous_distributions.pdf)

#### 2. Phitter Online Interactive Documentation
<div>
    <a href="https://phitter.io/distributions/continuous/alpha" target="_blank">• ALPHA</a>
    <a href="https://phitter.io/distributions/continuous/arcsine" target="_blank">• ARCSINE</a>
    <a href="https://phitter.io/distributions/continuous/argus" target="_blank">• ARGUS</a>
    <a href="https://phitter.io/distributions/continuous/beta" target="_blank">• BETA</a>
    <a href="https://phitter.io/distributions/continuous/beta_prime" target="_blank">• BETA PRIME</a>
    <a href="https://phitter.io/distributions/continuous/beta_prime_4p" target="_blank">• BETA PRIME 4P</a>
    <a href="https://phitter.io/distributions/continuous/bradford" target="_blank">• BRADFORD</a>
    <a href="https://phitter.io/distributions/continuous/burr" target="_blank">• BURR</a>
    <a href="https://phitter.io/distributions/continuous/burr_4p" target="_blank">• BURR 4P</a>
    <a href="https://phitter.io/distributions/continuous/cauchy" target="_blank">• CAUCHY</a>
    <a href="https://phitter.io/distributions/continuous/chi_square" target="_blank">• CHI SQUARE</a>
    <a href="https://phitter.io/distributions/continuous/chi_square_3p" target="_blank">• CHI SQUARE 3P</a>
    <a href="https://phitter.io/distributions/continuous/dagum" target="_blank">• DAGUM</a>
    <a href="https://phitter.io/distributions/continuous/dagum_4p" target="_blank">• DAGUM 4P</a>
    <a href="https://phitter.io/distributions/continuous/erlang" target="_blank">• ERLANG</a>
    <a href="https://phitter.io/distributions/continuous/erlang_3p" target="_blank">• ERLANG 3P</a>
    <a href="https://phitter.io/distributions/continuous/error_function" target="_blank">• ERROR FUNCTION</a>
    <a href="https://phitter.io/distributions/continuous/exponential" target="_blank">• EXPONENTIAL</a>
    <a href="https://phitter.io/distributions/continuous/exponential_2p" target="_blank">• EXPONENTIAL 2P</a>
    <a href="https://phitter.io/distributions/continuous/f" target="_blank">• F</a>
    <a href="https://phitter.io/distributions/continuous/fatigue_life" target="_blank">• FATIGUE LIFE</a>
    <a href="https://phitter.io/distributions/continuous/folded_normal" target="_blank">• FOLDED NORMAL</a>
    <a href="https://phitter.io/distributions/continuous/frechet" target="_blank">• FRECHET</a>
    <a href="https://phitter.io/distributions/continuous/f_4p" target="_blank">• F 4P</a>
    <a href="https://phitter.io/distributions/continuous/gamma" target="_blank">• GAMMA</a>
    <a href="https://phitter.io/distributions/continuous/gamma_3p" target="_blank">• GAMMA 3P</a>
    <a href="https://phitter.io/distributions/continuous/generalized_extreme_value" target="_blank">• GENERALIZED EXTREME VALUE</a>
    <a href="https://phitter.io/distributions/continuous/generalized_gamma" target="_blank">• GENERALIZED GAMMA</a>
    <a href="https://phitter.io/distributions/continuous/generalized_gamma_4p" target="_blank">• GENERALIZED GAMMA 4P</a>
    <a href="https://phitter.io/distributions/continuous/generalized_logistic" target="_blank">• GENERALIZED LOGISTIC</a>
    <a href="https://phitter.io/distributions/continuous/generalized_normal" target="_blank">• GENERALIZED NORMAL</a>
    <a href="https://phitter.io/distributions/continuous/generalized_pareto" target="_blank">• GENERALIZED PARETO</a>
    <a href="https://phitter.io/distributions/continuous/gibrat" target="_blank">• GIBRAT</a>
    <a href="https://phitter.io/distributions/continuous/gumbel_left" target="_blank">• GUMBEL LEFT</a>
    <a href="https://phitter.io/distributions/continuous/gumbel_right" target="_blank">• GUMBEL RIGHT</a>
    <a href="https://phitter.io/distributions/continuous/half_normal" target="_blank">• HALF NORMAL</a>
    <a href="https://phitter.io/distributions/continuous/hyperbolic_secant" target="_blank">• HYPERBOLIC SECANT</a>
    <a href="https://phitter.io/distributions/continuous/inverse_gamma" target="_blank">• INVERSE GAMMA</a>
    <a href="https://phitter.io/distributions/continuous/inverse_gamma_3p" target="_blank">• INVERSE GAMMA 3P</a>
    <a href="https://phitter.io/distributions/continuous/inverse_gaussian" target="_blank">• INVERSE GAUSSIAN</a>
    <a href="https://phitter.io/distributions/continuous/inverse_gaussian_3p" target="_blank">• INVERSE GAUSSIAN 3P</a>
    <a href="https://phitter.io/distributions/continuous/johnson_sb" target="_blank">• JOHNSON SB</a>
    <a href="https://phitter.io/distributions/continuous/johnson_su" target="_blank">• JOHNSON SU</a>
    <a href="https://phitter.io/distributions/continuous/kumaraswamy" target="_blank">• KUMARASWAMY</a>
    <a href="https://phitter.io/distributions/continuous/laplace" target="_blank">• LAPLACE</a>
    <a href="https://phitter.io/distributions/continuous/levy" target="_blank">• LEVY</a>
    <a href="https://phitter.io/distributions/continuous/loggamma" target="_blank">• LOGGAMMA</a>
    <a href="https://phitter.io/distributions/continuous/logistic" target="_blank">• LOGISTIC</a>
    <a href="https://phitter.io/distributions/continuous/loglogistic" target="_blank">• LOGLOGISTIC</a>
    <a href="https://phitter.io/distributions/continuous/loglogistic_3p" target="_blank">• LOGLOGISTIC 3P</a>
    <a href="https://phitter.io/distributions/continuous/lognormal" target="_blank">• LOGNORMAL</a>
    <a href="https://phitter.io/distributions/continuous/maxwell" target="_blank">• MAXWELL</a>
    <a href="https://phitter.io/distributions/continuous/moyal" target="_blank">• MOYAL</a>
    <a href="https://phitter.io/distributions/continuous/nakagami" target="_blank">• NAKAGAMI</a>
    <a href="https://phitter.io/distributions/continuous/non_central_chi_square" target="_blank">• NON CENTRAL CHI SQUARE</a>
    <a href="https://phitter.io/distributions/continuous/non_central_f" target="_blank">• NON CENTRAL F</a>
    <a href="https://phitter.io/distributions/continuous/non_central_t_student" target="_blank">• NON CENTRAL T STUDENT</a>
    <a href="https://phitter.io/distributions/continuous/normal" target="_blank">• NORMAL</a>
    <a href="https://phitter.io/distributions/continuous/pareto_first_kind" target="_blank">• PARETO FIRST KIND</a>
    <a href="https://phitter.io/distributions/continuous/pareto_second_kind" target="_blank">• PARETO SECOND KIND</a>
    <a href="https://phitter.io/distributions/continuous/pert" target="_blank">• PERT</a>
    <a href="https://phitter.io/distributions/continuous/power_function" target="_blank">• POWER FUNCTION</a>
    <a href="https://phitter.io/distributions/continuous/rayleigh" target="_blank">• RAYLEIGH</a>
    <a href="https://phitter.io/distributions/continuous/reciprocal" target="_blank">• RECIPROCAL</a>
    <a href="https://phitter.io/distributions/continuous/rice" target="_blank">• RICE</a>
    <a href="https://phitter.io/distributions/continuous/semicircular" target="_blank">• SEMICIRCULAR</a>
    <a href="https://phitter.io/distributions/continuous/trapezoidal" target="_blank">• TRAPEZOIDAL</a>
    <a href="https://phitter.io/distributions/continuous/triangular" target="_blank">• TRIANGULAR</a>
    <a href="https://phitter.io/distributions/continuous/t_student" target="_blank">• T STUDENT</a>
    <a href="https://phitter.io/distributions/continuous/t_student_3p" target="_blank">• T STUDENT 3P</a>
    <a href="https://phitter.io/distributions/continuous/uniform" target="_blank">• UNIFORM</a>
    <a href="https://phitter.io/distributions/continuous/weibull" target="_blank">• WEIBULL</a>
    <a href="https://phitter.io/distributions/continuous/weibull_3p" target="_blank">• WEIBULL 3P</a>
</div>

## Discrete Distributions
#### [1. PDF File Documentation Discrete Distributions](https://github.com/phitterio/phitter-kernel/blob/main/distributions_documentation/discrete/document_discrete_distributions/phitter_discrete_distributions.pdf)

#### 2. Phitter Online Interactive Documentation
<div style="display: flex; flex-wrap: wrap">
    <a href="https://phitter.io/distributions/discrete/bernoulli" target="_blank">• BERNOULLI</a>
    <a href="https://phitter.io/distributions/discrete/binomial" target="_blank">• BINOMIAL</a>
    <a href="https://phitter.io/distributions/discrete/geometric" target="_blank">• GEOMETRIC</a>
    <a href="https://phitter.io/distributions/discrete/hypergeometric" target="_blank">• HYPERGEOMETRIC</a>
    <a href="https://phitter.io/distributions/discrete/logarithmic" target="_blank">• LOGARITHMIC</a>
    <a href="https://phitter.io/distributions/discrete/negative_binomial" target="_blank">• NEGATIVE BINOMIAL</a>
    <a href="https://phitter.io/distributions/discrete/poisson" target="_blank">• POISSON</a>
    <a href="https://phitter.io/distributions/discrete/uniform" target="_blank">• UNIFORM</a>
</div>

