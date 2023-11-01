jStat = require("../node_modules/jstat");

dists = {
    generalized_pareto: {
        measurements: {
            nonCentralMoments: function (k, c, mu, sigma) {
                return undefined;
            },
            centralMoments: function (k, c, mu, sigma) {
                return undefined;
            },
            stats: {
                mean: function (c, mu, sigma) {
                    return mu + sigma / (1 - c);
                },
                variance: function (c, mu, sigma) {
                    return sigma * sigma / ((1 - c) * (1 - c) * (1 - 2 * c));
                },
                standardDeviation: function (c, mu, sigma) {
                    return Math.sqrt(this.variance(c, mu, sigma));
                },
                skewness: function (c, mu, sigma) {
                    return (2 * (1 + c) * Math.sqrt(1 - 2 * c)) / (1 - 3 * c);
                },
                kurtosis: function (c, mu, sigma) {
                    return (3 * (1 - 2 * c) * (2 * c * c + c + 3)) / ((1 - 3 * c) * (1 - 4 * c));
                },
                median: function (c, mu, sigma) {
                    return dists.generalized_pareto.measurements.ppf(0.5, c, mu, sigma);
                },
                mode: function (c, mu, sigma) {
                    return mu;
                },
            },
        }
    }
}
console.log(dists.generalized_pareto.measurements.stats.mean(-0.5, 10, 2))
console.log(dists.generalized_pareto.measurements.stats.variance(-0.5, 10, 2))
console.log(dists.generalized_pareto.measurements.stats.standardDeviation(-0.5, 10, 2))
console.log(dists.generalized_pareto.measurements.stats.skewness(-0.5, 10, 2))
console.log(dists.generalized_pareto.measurements.stats.kurtosis(-0.5, 10, 2))
// console.log(dists.generalized_pareto.measurements.stats.median(-0.5, 10, 2))
console.log(dists.generalized_pareto.measurements.stats.mode(-0.5, 10, 2))

// mean_value: 11.333333333333334
// variance_value: 0.8888888888888888
// standard_deviation_value: 0.9428090415820634
// skewness_value: 0.565685424949238
// kurtosis_value: 2.4
// median_value: 11.17157287525381
// mode_value: 10