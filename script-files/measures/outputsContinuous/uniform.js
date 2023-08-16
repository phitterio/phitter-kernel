jStat = require("../node_modules/jstat");

dists = {
    uniform: {
        measurements: {
            nonCentralMoments: function (k, a, b) {
                return undefined;
            },
            centralMoments: function (k, a, b) {
                return undefined;
            },
            stats: {
                mean: function (a, b) {
                    return (a + b) / 2;
                },
                variance: function (a, b) {
                    return (b - a) ** 2 / 12;
                },
                standardDeviation: function (a, b) {
                    return Math.sqrt(this.variance(a, b));
                },
                skewness: function (a, b) {
                    return 0;
                },
                kurtosis: function (a, b) {
                    return 3 - 6 / 5;
                },
                median: function (a, b) {
                    return dists.uniform.measurements.ppf(0.5, a, b);
                },
                mode: function (a, b) {
                    return undefined;
                },
            },
        }
    }
}
console.log(dists.uniform.measurements.stats.mean(50, 300))
console.log(dists.uniform.measurements.stats.variance(50, 300))
console.log(dists.uniform.measurements.stats.standardDeviation(50, 300))
console.log(dists.uniform.measurements.stats.skewness(50, 300))
console.log(dists.uniform.measurements.stats.kurtosis(50, 300))
// console.log(dists.uniform.measurements.stats.median(50, 300))
console.log(dists.uniform.measurements.stats.mode(50, 300))

// mean_value: 175
// variance_value: 5208.333333333333
// standard_deviation_value: 72.16878364870321
// skewness_value: 0
// kurtosis_value: 1.8
// median_value: 175
// mode_value: -