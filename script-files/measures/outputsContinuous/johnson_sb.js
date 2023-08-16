jStat = require("../node_modules/jstat");

dists = {
    johnson_sb: {
        measurements: {
            nonCentralMoments: function (k, xi, lambda, gamma, delta) {
                return undefined;
            },
            centralMoments: function (k, xi, lambda, gamma, delta) {
                return undefined;
            },
            stats: {
                mean: function (xi, lambda, gamma, delta) {
                    return undefined;
                },
                variance: function (xi, lambda, gamma, delta) {
                    return undefined;
                },
                standardDeviation: function (xi, lambda, gamma, delta) {
                    return Math.sqrt(this.variance(xi, lambda, gamma, delta));
                },
                skewness: function (xi, lambda, gamma, delta) {
                    return undefined;
                },
                kurtosis: function (xi, lambda, gamma, delta) {
                    return undefined;
                },
                median: function (xi, lambda, gamma, delta) {
                    return dists.johnson_sb.measurements.ppf(0.5, xi, lambda, gamma, delta);
                },
                mode: function (xi, lambda, gamma, delta) {
                    return undefined;
                },
            },
        }
    }
}
console.log(dists.johnson_sb.measurements.stats.mean(100, 1000, 5, 2))
console.log(dists.johnson_sb.measurements.stats.variance(100, 1000, 5, 2))
console.log(dists.johnson_sb.measurements.stats.standardDeviation(100, 1000, 5, 2))
console.log(dists.johnson_sb.measurements.stats.skewness(100, 1000, 5, 2))
console.log(dists.johnson_sb.measurements.stats.kurtosis(100, 1000, 5, 2))
// console.log(dists.johnson_sb.measurements.stats.median(100, 1000, 5, 2))
console.log(dists.johnson_sb.measurements.stats.mode(100, 1000, 5, 2))

// mean_value: complicated
// variance_value: complicated
// standard_deviation_value: complicated
// skewness_value: complicated
// kurtosis_value: complicated
// median_value: 175.85818002124356
// mode_value: complicated