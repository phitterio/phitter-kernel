jStat = require("../node_modules/jstat");

dists = {
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
                    return Math.sqrt(this.variance(n, p));
                },
                skewness: function (n, p) {
                    return (1 - p - p) / Math.sqrt(n * p * (1 - p));
                },
                kurtosis: function (n, p) {
                    return (1 - 6 * p * (1 - p)) / (n * p * (1 - p)) + 3;
                },
                median: function (n, p) {
                    return dists.binomial.measurements.ppf(0.5, n, p);
                },
                mode: function (n, p) {
                    return undefined;
                },
            },
        }
    }
}
console.log(dists.binomial.measurements.stats.mean(10.0, 0.7));
console.log(dists.binomial.measurements.stats.variance(10.0, 0.7));
console.log(dists.binomial.measurements.stats.standardDeviation(10.0, 0.7));
console.log(dists.binomial.measurements.stats.skewness(10.0, 0.7));
console.log(dists.binomial.measurements.stats.kurtosis(10.0, 0.7));
// console.log(dists.binomial.measurements.stats.median(10.0, 0.7))
console.log(dists.binomial.measurements.stats.mode(10.0, 0.7));

// mean_value: 7
// variance_value: 2.1
// standard_deviation_value: 1.449137675
// skewness_value: -0.2760262237
// kurtosis_value: 2.876190476
// median_value: 7
// mode_value: None
