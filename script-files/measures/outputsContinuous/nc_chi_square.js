jStat = require("../node_modules/jstat");

dists = {
    NON_CENTRAL_CHI_SQUARE: {
        measurements: {
            nonCentralMoments: function (k, lambda, n) {
                return undefined;
            },
            centralMoments: function (k, lambda, n) {
                return undefined;
            },
            stats: {
                mean: function (lambda, n) {
                    return lambda + n;
                },
                variance: function (lambda, n) {
                    return 2 * (n + 2 * lambda);
                },
                standardDeviation: function (lambda, n) {
                    return Math.sqrt(this.variance(lambda, n));
                },
                skewness: function (lambda, n) {
                    return 2 ** 1.5 * (n + 3 * lambda) / ((n + 2 * lambda) ** 1.5);
                },
                kurtosis: function (lambda, n) {
                    return 3 + 12 * (n + 4 * lambda) / ((n + 2 * lambda) ** 2);
                },
                median: function (lambda, n) {
                    return dists.NON_CENTRAL_CHI_SQUARE.measurements.ppf(0.5, lambda, n);
                },
                mode: function (lambda, n) {
                    return undefined;
                },
            },
        }
    }
}
console.log(dists.NON_CENTRAL_CHI_SQUARE.measurements.stats.mean(100, 5))
console.log(dists.NON_CENTRAL_CHI_SQUARE.measurements.stats.variance(100, 5))
console.log(dists.NON_CENTRAL_CHI_SQUARE.measurements.stats.standardDeviation(100, 5))
console.log(dists.NON_CENTRAL_CHI_SQUARE.measurements.stats.skewness(100, 5))
console.log(dists.NON_CENTRAL_CHI_SQUARE.measurements.stats.kurtosis(100, 5))
// console.log(dists.NON_CENTRAL_CHI_SQUARE.measurements.stats.median(100, 5))
console.log(dists.NON_CENTRAL_CHI_SQUARE.measurements.stats.mode(100, 5))

// mean_value: 105
// variance_value: 410
// standard_deviation_value: 20.248456731316587
// skewness_value: 0.2939097824176826
// kurtosis_value: 3.115645449137418
// median_value: None
// mode_value: None