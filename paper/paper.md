---
title: "Phitter: A Python Library for Probability Distribution Fitting and Analysis"
tags:
    - Python
    - Statistics
    - Probability Distributions
    - Data Analysis
    - Machine Learning
    - Simulation
    - Monte Carlo
authors:
    - name: Sebastián José Herrera Monterrosa
      orcid: 0009-0002-2766-642X
      affiliation: 1
    - name: Carlos Andrés Masmela Pinilla
      orcid: 0009-0000-0390-1558
      affiliation: 1
affiliations:
    - name: "Pontificia Universidad Javeriana"
      index: 1
date: 26 March 2024
bibliography: paper.bib
---

# Summary

Phitter is an open-source Python library that streamlines the process of fitting and analyzing probability distributions across various domains—including statistics, data science, operations research, and machine learning—by offering a comprehensive catalog of over 80 continuous and discrete distributions, multiple goodness-of-fit measures (Chi-Square, Kolmogorov-Smirnov, and Anderson-Darling), interactive visualizations for exploratory data analysis and model validation, and detailed modeling guides and spreadsheets to assist users in applying the chosen distribution for simulation or predictive tasks, thus reducing the complexity of distribution fitting and helping researchers and practitioners identify distributions that most closely model their data.

# Statement of Need

Fitting probability distributions to empirical data is a fundamental task in various scientific and engineering disciplines, crucial for applications such as stochastic modeling, risk assessment, and event simulation [@vose2008risk]. While methods exist for fitting specific distributions where parameters are easily derived from sample statistics (e.g., the normal distribution), the process becomes significantly more complex when aiming to identify the best-fitting distribution among a large set of candidates or when dealing with distributions requiring non-trivial parameter estimation techniques.

Although commercial software packages exist that automate distribution fitting, their commercial nature and cost limit accessibility for many researchers and practitioners. Within the open-source scientific Python ecosystem, the `scipy.stats` module [@2020SciPy-NMeth] offers a powerful foundation, providing implementations for numerous probability distributions and parameter estimation capabilities.

Therefore, there is a need for an accessible, open-source Python library specifically designed for automated probability distribution fitting that:

1.  Provides implementations for a wide range of common distributions.
2.  Implements efficient parameter estimation techniques, prioritizing th time estimation.
3.  Provide an easy-to-use interface for fitting distributions and analyzing and visualizing results.

Phitter has been developed to address these needs, providing the scientific community with a dedicated tool that simplifies and standardizes the process of probability distribution fitting, bridging the gap left by existing general-purpose statistical libraries and proprietary software.

# Key Features

1. Phitter implements probability distributions as standardized as possible according to the following sources: [@walck1996hand], [@mclaughlin2001compendium], [@wiki:List_of_probability_distributions]

2. **Accelerated Parameter Estimation:** For a subset of implemented distributions, Phitter utilizes direct solving of the distribution's parametric equation system. This approach offers significant computational speed advantages compared to iterative methods like standard Maximum Likelihood Estimation (MLE), particularly beneficial for large datasets. Performance benchmarks detailing estimation times are provided [Estimation Time Parameters for Continuous Distributions](https://docs-phitter-kernel.netlify.app/documentation/benchmarks/continuous/continuous-parameters-estimation.html). Where direct parametric solutions are not implemented or feasible, Phitter seamlessly integrates with and leverages the well-established MLE implementation provided by the SciPy library (`scipy.stats.rv_continuous.fit` and related methods). This ensures broad applicability across a wide range of distributions while prioritizing speed where possible.

3. **Anderson Darling:** Phitter evaluetes de Anderson Darling distribution according this article [@marsaglia2004evaluating].

4. **Comprehensive Distribution Documentation:** Phitter is accompanied by detailed documentation for both continuous and discrete distributions [Continuous Distributions](https://docs-phitter-kernel.netlify.app/documentation/distributions/continuous-distributions.html) and [Discrete Distributions](https://docs-phitter-kernel.netlify.app/documentation/distributions/discrete-distributions.html). This documentation outlines the mathematical formulations used and provides Excel and Google Sheet implemetation for each distribution.

5. **Fit characteristics:**

-   **Extensive Distribution Library:** Access to over 80 continuous and discrete probability distributions.
-   **Multiple Goodness-of-Fit Tests:** Offers a choice of Kolmogorov-Smirnov (K-S), Chi-Square, and Anderson-Darling (A-D) tests for quantitative fit evaluation.
-   **Parallel Processing:** Utilizes multiprocessing for faster evaluation of multiple distributions, particularly effective for large datasets (e.g., 100K+ samples).
-   **Integrated Visualizations:** Provides built-in plotting functions (Histogram/PDF overlay, ECDF vs. CDF, Q-Q plots) for visual assessment of distribution fits.
-   **Automated Modeling Guides:** Generates detailed reports for best-fit distributions, including parameters, key formulas (PDF, CDF, PPF), usage recommendations, and implementation details.

6. **Simulation**: Phitter not only incorporates a robust set of functionalities for fitting and analyzing over 80 probability distributions, both continuous and discrete, but also offers capabilities for simulating processes and queues: FIFO, LIFO and PBS.

# Comparison with Existing Tools

-   **distfit**  
    distfit iterates over some or all scipy distributions. With large samples, it is significantly slower. It does not support parallelism.

-   **fitter**  
    fitter iterates over some or all scipy distributions. With large samples, it is significantly slower. It supports parallelism.

# Documentation

Find the complete Phitter documentation [here](https://docs-phitter-kernel.netlify.app/).

# References
