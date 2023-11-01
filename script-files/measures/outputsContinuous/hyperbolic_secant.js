jStat = require("../node_modules/jstat");

dists = {
    hyperbolic_secant: {
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
                    return sigma ** 2;
                },
                standardDeviation: function (mu, sigma) {
                    return Math.sqrt(this.variance(mu, sigma));
                },
                skewness: function (mu, sigma) {
                    return 0;
                },
                kurtosis: function (mu, sigma) {
                    return 5;
                },
                median: function (mu, sigma) {
                    return dists.hyperbolic_secant.measurements.ppf(0.5, mu, sigma);
                },
                mode: function (mu, sigma) {
                    return mu;
                },
            },
        }
    }
}
console.log(dists.hyperbolic_secant.measurements.stats.mean(3, 5))
console.log(dists.hyperbolic_secant.measurements.stats.variance(3, 5))
console.log(dists.hyperbolic_secant.measurements.stats.standardDeviation(3, 5))
console.log(dists.hyperbolic_secant.measurements.stats.skewness(3, 5))
console.log(dists.hyperbolic_secant.measurements.stats.kurtosis(3, 5))
// console.log(dists.hyperbolic_secant.measurements.stats.median(3, 5))
console.log(dists.hyperbolic_secant.measurements.stats.mode(3, 5))

// mean_value: 3
// variance_value: 25
// standard_deviation_value: 5
// skewness_value: 0
// kurtosis_value: 5
// median_value: 3
// mode_value: 3