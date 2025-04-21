* [phitter](modules.html)
* [phitter package](phitter.html)
* [phitter.simulation package](phitter.simulation.html)
* phitter.simulation.process\_simulation package
* [View page source](_sources/phitter.simulation.process_simulation.rst.txt)

---

# phitter.simulation.process\_simulation package[](#phitter-simulation-process-simulation-package "Link to this heading")

## Submodules[](#submodules "Link to this heading")

## phitter.simulation.process\_simulation.process\_simulation module[](#module-phitter.simulation.process_simulation.process_simulation "Link to this heading")

*class* phitter.simulation.process\_simulation.process\_simulation.ProcessSimulation[](#phitter.simulation.process_simulation.process_simulation.ProcessSimulation "Link to this definition")
:   Bases: `object`

## add\_process(*prob\_distribution*, *parameters*, *process\_id*, *number\_of\_products=1*, *number\_of\_servers=1*, *new\_branch=False*, *previous\_ids=None*)[](#phitter.simulation.process_simulation.process_simulation.ProcessSimulation.add_process "Link to this definition")
Add element to the simulation

#### Parameters
* **prob\_distribution** (*str*) – Probability distribution to be used. You can use one of the following: ‘alpha’, ‘arcsine’, ‘argus’, ‘beta’, ‘beta\_prime’, ‘beta\_prime\_4p’, ‘bradford’, ‘burr’, ‘burr\_4p’, ‘cauchy’, ‘chi\_square’, ‘chi\_square\_3p’, ‘dagum’, ‘dagum\_4p’, ‘erlang’, ‘erlang\_3p’, ‘error\_function’, ‘exponential’, ‘exponential\_2p’, ‘f’, ‘fatigue\_life’, ‘folded\_normal’, ‘frechet’, ‘f\_4p’, ‘gamma’, ‘gamma\_3p’, ‘generalized\_extreme\_value’, ‘generalized\_gamma’, ‘generalized\_gamma\_4p’, ‘generalized\_logistic’, ‘generalized\_normal’, ‘generalized\_pareto’, ‘gibrat’, ‘gumbel\_left’, ‘gumbel\_right’, ‘half\_normal’, ‘hyperbolic\_secant’, ‘inverse\_gamma’, ‘inverse\_gamma\_3p’, ‘inverse\_gaussian’, ‘inverse\_gaussian\_3p’, ‘johnson\_sb’, ‘johnson\_su’, ‘kumaraswamy’, ‘laplace’, ‘levy’, ‘loggamma’, ‘logistic’, ‘loglogistic’, ‘loglogistic\_3p’, ‘lognormal’, ‘maxwell’, ‘moyal’, ‘nakagami’, ‘non\_central\_chi\_square’, ‘non\_central\_f’, ‘non\_central\_t\_student’, ‘normal’, ‘pareto\_first\_kind’, ‘pareto\_second\_kind’, ‘pert’, ‘power\_function’, ‘rayleigh’, ‘reciprocal’, ‘rice’, ‘semicircular’, ‘trapezoidal’, ‘triangular’, ‘t\_student’, ‘t\_student\_3p’, ‘uniform’, ‘weibull’, ‘weibull\_3p’, ‘bernoulli’, ‘binomial’, ‘geometric’, ‘hypergeometric’, ‘logarithmic’, ‘negative\_binomial’, ‘poisson’.
- * **parameters** (*dict*) – Parameters of the probability distribution.
- * **process\_id** (*str*) – Unique name of the process to be simulated
- * **number\_of\_products** (*int* *,**optional*) – Number of elements that are needed to simulate in that stage. Value has to be greater than 0. Defaults equals to 1.
- * **number\_of\_servers** (*int**,* *optional*) – Number of servers that process has and are needed to simulate in that stage. Value has to be greater than 0. Defaults equals to 1.
- * **new\_branch** (*bool* *,**optional*) – Required if you want to start a new process that does not have previous processes. You cannot use this parameter at the same time with “previous\_id”. Defaults to False.
- * **previous\_id** (*list**[**str**]**,* *optional*) – Required if you have previous processes that are before this process. You cannot use this parameter at the same time with “new\_branch”. Defaults to None.

#### Return type
`None`

## process\_graph(*graph\_direction='LR'*, *save\_graph\_pdf=False*)[](#phitter.simulation.process_simulation.process_simulation.ProcessSimulation.process_graph "Link to this definition")
Generates the graph of the process

#### Parameters
* **graph\_direction** (*str**,* *optional*) – You can show the graph in two ways: ‘LR’ left to right OR ‘TB’ top to bottom. Defaults to ‘LR’.
- * **save\_graph\_pdf** (*bool**,* *optional*) – You can save the process graph in a PDF file. Defaults to False.

#### Return type
`None`

## run(*number\_of\_simulations=1*)[](#phitter.simulation.process_simulation.process_simulation.ProcessSimulation.run "Link to this definition")
Simulation of the described process

#### Parameters
**number\_of\_simulations** (*int**,* *optional*) – Number of simulations of the process that you want to do. Defaults to 1.

#### Returns
Results of every simulation requested

#### Return type
list[float]

## run\_confidence\_interval(*confidence\_level=0.95*, *number\_of\_simulations=1*, *replications=30*)[](#phitter.simulation.process_simulation.process_simulation.ProcessSimulation.run_confidence_interval "Link to this definition")
Generates a confidence interval for the replications of the requested number of simulations.

#### Parameters
* **confidence\_level** (*float**,* *optional*) – Confidence required of the interval. Defaults to 0.95.
- * **number\_of\_simulations** (*int**,* *optional*) – Number of simulations that are going to be run in each replication. Defaults to 1.
- * **replications** (*int**,* *optional*) – Number of samples needed. Defaults to 30.

#### Returns
Returns the lower bound, average, upper bound and standard deviation of the confidence interval

#### Return type
tuple[float]

## simulation\_metrics()[](#phitter.simulation.process_simulation.process_simulation.ProcessSimulation.simulation_metrics "Link to this definition")
Here you can find the average time per process and standard deviation

#### Returns
Average and Standard deviation

#### Return type
pd.DataFrame

## Module contents[](#module-phitter.simulation.process_simulation "Link to this heading")

[Previous](phitter.simulation.own_distribution.html "phitter.simulation.own_distribution package")
[Next](phitter.simulation.queueing_simulation.html "phitter.simulation.queueing_simulation package")

---

© Copyright .

Built with [Sphinx](https://www.sphinx-doc.org/) using a
[theme](https://github.com/readthedocs/sphinx_rtd_theme)
provided by [Read the Docs](https://readthedocs.org).