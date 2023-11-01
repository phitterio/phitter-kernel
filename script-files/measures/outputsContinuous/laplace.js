jStat = require("../node_modules/jstat");

dists = {
    laplace: {
        measurements: {
            nonCentralMoments: function (k, mu, b) {
                return undefined;
            },
            centralMoments: function (k, mu, b) {
                return undefined;
            },
            stats: {
                mean: function (mu, b) {
                    return mu;
                },
                variance: function (mu, b) {
                    return 2 * b ** 2;
                },
                standardDeviation: function (mu, b) {
                    return Math.sqrt(this.variance(mu, b));
                },
                skewness: function (mu, b) {
                    return 0;
                },
                kurtosis: function (mu, b) {
                    return 6;
                },
                median: function (mu, b) {
                    return dists.laplace.measurements.ppf(0.5, mu, b);
                },
                mode: function (mu, b) {
                    return mu;
                },
            },
        }
    }
}
console.log(dists.laplace.measurements.stats.mean(17, 5))
console.log(dists.laplace.measurements.stats.variance(17, 5))
console.log(dists.laplace.measurements.stats.standardDeviation(17, 5))
console.log(dists.laplace.measurements.stats.skewness(17, 5))
console.log(dists.laplace.measurements.stats.kurtosis(17, 5))
// console.log(dists.laplace.measurements.stats.median(17, 5))
console.log(dists.laplace.measurements.stats.mode(17, 5))

// mean_value: 17
// variance_value: 50
// standard_deviation_value: 7.0710678118654755
// skewness_value: 0
// kurtosis_value: 6
// median_value: 17
// mode_value: 17