jStat = require("../node_modules/jstat");

dists = {
    logistic: {
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
                    return sigma * sigma * Math.PI * Math.PI / 3;
                },
                standardDeviation: function (mu, sigma) {
                    return Math.sqrt(this.variance(mu, sigma));
                },
                skewness: function (mu, sigma) {
                    return 0;
                },
                kurtosis: function (mu, sigma) {
                    return 4.2;
                },
                median: function (mu, sigma) {
                    return dists.logistic.measurements.ppf(0.5, mu, sigma);
                },
                mode: function (mu, sigma) {
                    return mu;
                },
            },
        }
    }
}
console.log(dists.logistic.measurements.stats.mean(10, 5))
console.log(dists.logistic.measurements.stats.variance(10, 5))
console.log(dists.logistic.measurements.stats.standardDeviation(10, 5))
console.log(dists.logistic.measurements.stats.skewness(10, 5))
console.log(dists.logistic.measurements.stats.kurtosis(10, 5))
// console.log(dists.logistic.measurements.stats.median(10, 5))
console.log(dists.logistic.measurements.stats.mode(10, 5))

// mean_value: 10
// variance_value: 82.24670334241132
// standard_deviation_value: 9.068996821171089
// skewness_value: 0
// kurtosis_value: 4.2
// median_value: 10
// mode_value: 10