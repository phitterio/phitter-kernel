jStat = require("../node_modules/jstat");

dists = {
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
                    return Math.sqrt(this.variance(lambda));
                },
                skewness: function (lambda) {
                    return lambda ** -0.5;
                },
                kurtosis: function (lambda) {
                    return 1 / lambda + 3;
                },
                median: function (lambda) {
                    return dists.poisson.measurements.ppf(0.5, lambda);
                },
                mode: function (lambda) {
                    return Math.floor(lambda);
                },
            },
        }
    }
}
console.log(dists.poisson.measurements.stats.mean(5));
console.log(dists.poisson.measurements.stats.variance(5));
console.log(dists.poisson.measurements.stats.standardDeviation(5));
console.log(dists.poisson.measurements.stats.skewness(5));
console.log(dists.poisson.measurements.stats.kurtosis(5));
// console.log(dists.poisson.measurements.stats.median(5))
console.log(dists.poisson.measurements.stats.mode(5));

// mean_value: 5
// variance_value: 5
// standard_deviation_value: 2.23606797749979
// skewness_value: 0.4472135954999579
// kurtosis_value: 3.2
// median_value: 5
// mode_value: 5
