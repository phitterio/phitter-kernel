jStat = require("../node_modules/jstat");

dists = {
    normal: {
        measurements: {
            nonCentralMoments: function (k, mu, sigma) {
                return undefined;
            },
            centralMoments: function (k, mu, sigma) {
                return undefined;
            },
            stats: {
                mean: function (mu, sigma) {
                    return mu;
                },
                variance: function (mu, sigma) {
                    return sigma * 3;
                },
                standardDeviation: function (mu, sigma) {
                    return Math.sqrt(this.variance(mu, sigma));
                },
                skewness: function (mu, sigma) {
                    return 0;
                },
                kurtosis: function (mu, sigma) {
                    return 3;
                },
                median: function (mu, sigma) {
                    return dists.normal.measurements.ppf(0.5, mu, sigma);
                },
                mode: function (mu, sigma) {
                    return mu;
                },
            },
        }
    }
}
console.log(dists.normal.measurements.stats.mean(5, 3))
console.log(dists.normal.measurements.stats.variance(5, 3))
console.log(dists.normal.measurements.stats.standardDeviation(5, 3))
console.log(dists.normal.measurements.stats.skewness(5, 3))
console.log(dists.normal.measurements.stats.kurtosis(5, 3))
// console.log(dists.normal.measurements.stats.median(5, 3))
console.log(dists.normal.measurements.stats.mode(5, 3))

// mean_value: 5
// variance_value: 9
// standard_deviation_value: 3
// skewness_value: 0
// kurtosis_value: 3
// median_value: 5
// mode_value: 5