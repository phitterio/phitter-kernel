jStat = require("../node_modules/jstat");

dists = {
    cauchy: {
        measurements: {
            nonCentralMoments: function (k, x0, gamma) {
                return undefined;
            },
            centralMoments: function (k, x0, gamma) {
                return undefined;
            },
            stats: {
                mean: function (x0, gamma) {
                    return undefined;
                },
                variance: function (x0, gamma) {
                    return undefined;
                },
                standardDeviation: function (x0, gamma) {
                    return Math.sqrt(this.variance(x0, gamma));
                },
                skewness: function (x0, gamma) {
                    return undefined;
                },
                kurtosis: function (x0, gamma) {
                    return undefined;
                },
                median: function (x0, gamma) {
                    return dists.cauchy.measurements.ppf(0.5, x0, gamma);
                },
                mode: function (x0, gamma) {
                    return x0;
                },
            },
        }
    }
}
console.log(dists.cauchy.measurements.stats.mean(5, 1))
console.log(dists.cauchy.measurements.stats.variance(5, 1))
console.log(dists.cauchy.measurements.stats.standardDeviation(5, 1))
console.log(dists.cauchy.measurements.stats.skewness(5, 1))
console.log(dists.cauchy.measurements.stats.kurtosis(5, 1))
// console.log(dists.cauchy.measurements.stats.median(5, 1))
console.log(dists.cauchy.measurements.stats.mode(5, 1))

// mean_value: undefined
// variance_value: undefined
// standard_deviation_value: undefined
// skewness_value: undefined
// kurtosis_value: undefined
// median_value: 5
// mode_value: 5