const discreteDistributionsMeasurements = {
    bernoulli: {
        measurements: {
            nonCentralMoments: function (k, p) {
                return undefined;
            },
            centralMoments: function (k, p) {
                return undefined;
            },
            stats: {
                mean: function (p) {
                    return p;
                },
                variance: function (p) {
                    return p * (1 - p);
                },
                standardDeviation: function (p) {
                    return this.variance(p) !== undefined ? numpy.sqrt(this.variance(p)!) : undefined;
                },
                skewness: function (p) {
                    return (1 - 2 * p) / numpy.sqrt(p * (1 - p));
                },
                kurtosis: function (p) {
                    return (6 * p * p - 6 * p + 1) / (p * (1 - p)) + 3;
                },
                median: function (p) {
                    return discreteDistributions.bernoulli.ppf(0.5, p);
                },
                mode: function (p) {
                    return p < 0.5 ? 0 : 1;
                },
            },
        },
    },
    binomial: {
        measurements: {
            nonCentralMoments: function (k, n, p) {
                return undefined;
            },
            centralMoments: function (k, n, p) {
                return undefined;
            },
            stats: {
                mean: function (n, p) {
                    return n * p;
                },
                variance: function (n, p) {
                    return n * p * (1 - p);
                },
                standardDeviation: function (n, p) {
                    return this.variance(n, p) !== undefined ? numpy.sqrt(this.variance(n, p)!) : undefined;
                },
                skewness: function (n, p) {
                    return (1 - p - p) / numpy.sqrt(n * p * (1 - p));
                },
                kurtosis: function (n, p) {
                    return (1 - 6 * p * (1 - p)) / (n * p * (1 - p)) + 3;
                },
                median: function (n, p) {
                    return discreteDistributions.binomial.ppf(0.5, n, p);
                },
                mode: function (n, p) {
                    return undefined;
                },
            },
        },
    },
    geometric: {
        measurements: {
            nonCentralMoments: function (k, p) {
                return undefined;
            },
            centralMoments: function (k, p) {
                return undefined;
            },
            stats: {
                mean: function (p) {
                    return 1 / p;
                },
                variance: function (p) {
                    return (1 - p) / (p * p);
                },
                standardDeviation: function (p) {
                    return this.variance(p) !== undefined ? numpy.sqrt(this.variance(p)!) : undefined;
                },
                skewness: function (p) {
                    return (2 - p) / numpy.sqrt(1 - p);
                },
                kurtosis: function (p) {
                    return 3 + 6 + (p * p) / (1 - p);
                },
                median: function (p) {
                    return discreteDistributions.geometric.ppf(0.5, p);
                },
                mode: function (p) {
                    return 1.0;
                },
            },
        },
    },
    hypergeometric: {
        measurements: {
            nonCentralMoments: function (k, N, K, n) {
                return undefined;
            },
            centralMoments: function (k, N, K, n) {
                return undefined;
            },
            stats: {
                mean: function (N, K, n) {
                    return (n * K) / N;
                },
                variance: function (N, K, n) {
                    return ((n * K) / N) * ((N - K) / N) * ((N - n) / (N - 1));
                },
                standardDeviation: function (N, K, n) {
                    return this.variance(N, K, n) !== undefined ? numpy.sqrt(this.variance(N, K, n)!) : undefined;
                },
                skewness: function (N, K, n) {
                    return ((N - 2 * K) * numpy.sqrt(N - 1) * (N - 2 * n)) / (numpy.sqrt(n * K * (N - K) * (N - n)) * (N - 2));
                },
                kurtosis: function (N, K, n) {
                    return (
                        3 +
                        (1 / (n * K * (N - K) * (N - n) * (N - 2) * (N - 3))) *
                            ((N - 1) * N * N * (N * (N + 1) - 6 * K * (N - K) - 6 * n * (N - n)) + 6 * n * K * (N - K) * (N - n) * (5 * N - 6))
                    );
                },
                median: function (N, K, n) {
                    return discreteDistributions.hypergeometric.ppf(0.5, N, K, n);
                },
                mode: function (N, K, n) {
                    return Math.floor(((n + 1) * (K + 1)) / (N + 2));
                },
            },
        },
    },
    logarithmic: {
        measurements: {
            nonCentralMoments: function (k, p) {
                return undefined;
            },
            centralMoments: function (k, p) {
                return undefined;
            },
            stats: {
                mean: function (p) {
                    return -p / ((1 - p) * numpy.log(1 - p));
                },
                variance: function (p) {
                    return (-p * (p + numpy.log(1 - p))) / ((1 - p) ** 2 * numpy.log(1 - p) ** 2);
                },
                standardDeviation: function (p) {
                    return this.variance(p) !== undefined ? numpy.sqrt(this.variance(p)!) : undefined;
                },
                skewness: function (p) {
                    return (
                        (-(2 * p ** 2 + 3 * p * numpy.log(1 - p) + (1 + p) * numpy.log(1 - p) ** 2) /
                            (numpy.log(1 - p) * (p + numpy.log(1 - p)) * numpy.sqrt(-p * (p + numpy.log(1 - p))))) *
                        numpy.log(1 - p)
                    );
                },
                kurtosis: function (p) {
                    return (
                        -(6 * p ** 3 + 12 * p ** 2 * numpy.log(1 - p) + p * (4 * p + 7) * numpy.log(1 - p) ** 2 + (p ** 2 + 4 * p + 1) * numpy.log(1 - p) ** 3) /
                        (p * (p + numpy.log(1 - p)) ** 2)
                    );
                },
                median: function (p) {
                    return discreteDistributions.logarithmic.ppf(0.5, p);
                },
                mode: function (p) {
                    return 1;
                },
            },
        },
    },
    negative_binomial: {
        measurements: {
            nonCentralMoments: function (k, r, p) {
                return undefined;
            },
            centralMoments: function (k, r, p) {
                return undefined;
            },
            stats: {
                mean: function (r, p) {
                    return (r * (1 - p)) / p;
                },
                variance: function (r, p) {
                    return (r * (1 - p)) / (p * p);
                },
                standardDeviation: function (r, p) {
                    return this.variance(r, p) !== undefined ? numpy.sqrt(this.variance(r, p)!) : undefined;
                },
                skewness: function (r, p) {
                    return (2 - p) / numpy.sqrt(r * (1 - p));
                },
                kurtosis: function (r, p) {
                    return 6 / r + (p * p) / (r * (1 - p)) + 3;
                },
                median: function (r, p) {
                    return discreteDistributions.negative_binomial.ppf(0.5, r, p);
                },
                mode: function (r, p) {
                    return Math.floor(((r - 1) * (1 - p)) / p, 0);
                },
            },
        },
    },
    poisson: {
        measurements: {
            nonCentralMoments: function (k, lambda) {
                return undefined;
            },
            centralMoments: function (k, lambda) {
                return undefined;
            },
            stats: {
                mean: function (lambda) {
                    return lambda;
                },
                variance: function (lambda) {
                    return lambda;
                },
                standardDeviation: function (lambda) {
                    return this.variance(lambda) !== undefined ? numpy.sqrt(this.variance(lambda)!) : undefined;
                },
                skewness: function (lambda) {
                    return lambda ** -0.5;
                },
                kurtosis: function (lambda) {
                    return 1 / lambda + 3;
                },
                median: function (lambda) {
                    return discreteDistributions.poisson.ppf(0.5, lambda);
                },
                mode: function (lambda) {
                    return Math.floor(lambda);
                },
            },
        },
    },
    uniform: {
        measurements: {
            nonCentralMoments: function (k, a, b) {
                return undefined;
            },
            centralMoments: function (k, a, b) {
                return undefined;
            },
            stats: {
                mean: function (a, b) {
                    return (a + b) / 2;
                },
                variance: function (a, b) {
                    return ((b - a + 1) * (b - a + 1) - 1) / 12;
                },
                standardDeviation: function (a, b) {
                    return this.variance(a, b) !== undefined ? numpy.sqrt(this.variance(a, b)!) : undefined;
                },
                skewness: function (a, b) {
                    return 0;
                },
                kurtosis: function (a, b) {
                    return ((-6 / 5) * ((b - a + 1) * (b - a + 1) + 1)) / ((b - a + 1) * (b - a + 1) - 1) + 3;
                },
                median: function (a, b) {
                    return discreteDistributions.uniform.ppf(0.5, a, b);
                },
                mode: function (a, b) {
                    return undefined;
                },
            },
        },
    },
};
