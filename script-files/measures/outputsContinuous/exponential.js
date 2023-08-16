jStat = require("../node_modules/jstat");

dists = {
    exponential: {
        measurements: {
            nonCentralMoments: function (k, lambda) {
                return undefined;
            },
            centralMoments: function (k, lambda) {
                return undefined;
            },
            stats: {
                mean: function (lambda) {
                    return 1 / lambda;
                },
                variance: function (lambda) {
                    return 1 / (lambda * lambda);
                },
                standardDeviation: function (lambda) {
                    return Math.sqrt(this.variance(lambda));
                },
                skewness: function (lambda) {
                    return 2;
                },
                kurtosis: function (lambda) {
                    return 9;
                },
                median: function (lambda) {
                    return dists.exponential.measurements.ppf(0.5, lambda);
                },
                mode: function (lambda) {
                    return 0;
                },
            },
        }
    }
}
console.log(dists.exponential.measurements.stats.mean(0.1))
console.log(dists.exponential.measurements.stats.variance(0.1))
console.log(dists.exponential.measurements.stats.standardDeviation(0.1))
console.log(dists.exponential.measurements.stats.skewness(0.1))
console.log(dists.exponential.measurements.stats.kurtosis(0.1))
// console.log(dists.exponential.measurements.stats.median(0.1))
console.log(dists.exponential.measurements.stats.mode(0.1))

// mean_value: 10
// variance_value: 99.99999999999999
// standard_deviation_value: 10
// skewness_value: 2
// kurtosis_value: 9
// median_value: 6.931471805599452
// mode_value: 0