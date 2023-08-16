jStat = require("../node_modules/jstat");

dists = {
    error_function: {
        measurements: {
            nonCentralMoments: function (k, h) {
                return undefined;
            },
            centralMoments: function (k, h) {
                return undefined;
            },
            stats: {
                mean: function (h) {
                    return 0;
                },
                variance: function (h) {
                    return 1 / (2 * h ** 2);
                },
                standardDeviation: function (h) {
                    return Math.sqrt(this.variance(h));
                },
                skewness: function (h) {
                    return 0;
                },
                kurtosis: function (h) {
                    return 3;
                },
                median: function (h) {
                    return dists.error_function.measurements.ppf(0.5, h);
                },
                mode: function (h) {
                    return 0;
                },
            },
        }
    }
}
console.log(dists.error_function.measurements.stats.mean(0.01))
console.log(dists.error_function.measurements.stats.variance(0.01))
console.log(dists.error_function.measurements.stats.standardDeviation(0.01))
console.log(dists.error_function.measurements.stats.skewness(0.01))
console.log(dists.error_function.measurements.stats.kurtosis(0.01))
// console.log(dists.error_function.measurements.stats.median(0.01))
console.log(dists.error_function.measurements.stats.mode(0.01))

// mean_value: 0
// variance_value: 5000
// standard_deviation_value: 70.71067811865476
// skewness_value: 0
// kurtosis_value: 3
// median_value: 0
// mode_value: 0