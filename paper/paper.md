---
title: "Phitter: A Python Library for Probability Distribution Fitting and Analysis"
tags:
    - Python
    - Statistics
    - probability distributions
    - data analysis
    - machine learning
    - simulation
    - monte carlo
authors:
    - name: Sebastián José Herrera Monterrosa
      orcid: 0009-0002-2766-642X
      affiliation: 1
affiliations:
    - name: Pontificia Universidad Javeriana
      index: 1
date: 26 March 2024
bibliography: paper.bib
---

# Summary

Phitter is a Python library designed to analyze datasets and determine the best analytical probability distributions that represent them. It provides a comprehensive suite of tools for fitting and analyzing over 80 probability distributions, both continuous and discrete. Phitter implements three goodness-of-fit tests and offers interactive visualizations to aid in the analysis process. For each selected probability distribution, Phitter provides a standard modeling guide along with detailed spreadsheets that outline the methodology for using the chosen distribution in various fields such as data science, operations research, and artificial intelligence.

# Statement of Need

In the fields of data science, statistics, and machine learning, understanding the underlying probability distributions of datasets is crucial for accurate modeling and prediction. However, identifying the most appropriate distribution for a given dataset can be a complex and time-consuming task. Phitter addresses this need by providing a user-friendly, efficient, and comprehensive tool for probability distribution fitting and analysis.

Phitter stands out from existing tools by offering:

1. A wide range of over 80 probability distributions, including both continuous and discrete options.
2. Implementation of multiple goodness-of-fit tests (Chi-Square, Kolmogorov-Smirnov, and Anderson-Darling).
3. Interactive visualizations for better understanding and interpretation of results.
4. Accelerated fitting capabilities for large datasets (over 100K samples).
5. Detailed modeling guides and spreadsheets for practical application in various fields.

# Features and Functionality

Phitter offers a range of features designed to streamline the process of probability distribution analysis:

-   **Flexible Fitting**: Users can fit both continuous and discrete distributions to their data.
-   **Customizable Analysis**: Options to specify the number of bins, confidence level, and distributions to fit.
-   **Parallel Processing**: Support for multi-threaded fitting to improve performance.
-   **Comprehensive Output**: Detailed summaries of fitted distributions, including parameters, test statistics, and rankings.
-   **Visualization Tools**: Functions to plot histograms, PDFs, ECDFs, and Q-Q plots for visual analysis.
-   **Distribution Utilities**: Methods to work with individual distributions, including CDF, PDF, PPF, and sampling functions.

# Implementation and Usage

Phitter is implemented in Python and is available via PyPI. It requires Python 3.9 or higher. The library can be easily installed using pip:

```
pip install phitter
```

Basic usage involves creating a `PHITTER` object with a dataset and calling the `fit()` method:

```python
import phitter

data = [...]  # Your dataset
phi = phitter.Phitter(data=data)
phi.fit()
```

More advanced usage allows for customization of fitting parameters and specific distribution analysis.

# Conclusion

Phitter provides researchers, data scientists, and statisticians with a powerful tool for probability distribution analysis. By offering a comprehensive set of distributions, multiple goodness-of-fit tests, and interactive visualizations, Phitter simplifies the process of identifying and working with probability distributions in various data-driven fields.

# References
