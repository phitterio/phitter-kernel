jStat = require("../node_modules/jstat");

dists = {
    hyperbolic_secant: {
        measurements: {
            nonCentralMoments: function (k, miu, sigma) {
                return undefined;
            },
            centralMoments: function (k, miu, sigma) {
                return undefined;
            },
            stats: {
                mean: function (miu, sigma) {
                    return miu;
                },
                variance: function (miu, sigma) {
                    return sigma ** 2;
                },
                standardDeviation: function (miu, sigma) {
                    return Math.sqrt(this.variance(miu, sigma));
                },
                skewness: function (miu, sigma) {
                    return 0;
                },
                kurtosis: function (miu, sigma) {
                    return 5;
                },
                median: function (miu, sigma) {
                    return dists.hyperbolic_secant.measurements.ppf(0.5, miu, sigma);
                },
                mode: function (miu, sigma) {
                    return miu;
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