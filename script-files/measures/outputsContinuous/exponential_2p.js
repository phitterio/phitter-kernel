jStat = require("../node_modules/jstat");

dists = {
    exponential_2p: {
        measurements: {
            nonCentralMoments: function (k, lambda, loc) {
                return undefined;
            },
            centralMoments: function (k, lambda, loc) {
                return undefined;
            },
            stats: {
                mean: function (lambda, loc) {
                    return 1 / lambda + loc;
                },
                variance: function (lambda, loc) {
                    return 1 / (lambda * lambda);
                },
                standardDeviation: function (lambda, loc) {
                    return Math.sqrt(this.variance(lambda, loc));
                },
                skewness: function (lambda, loc) {
                    return 2;
                },
                kurtosis: function (lambda, loc) {
                    return 9;
                },
                median: function (lambda, loc) {
                    return dists.exponential_2p.measurements.ppf(0.5, lambda, loc);
                },
                mode: function (lambda, loc) {
                    return loc;
                },
            },
        }
    }
}
console.log(dists.exponential_2p.measurements.stats.mean(0.1, 100))
console.log(dists.exponential_2p.measurements.stats.variance(0.1, 100))
console.log(dists.exponential_2p.measurements.stats.standardDeviation(0.1, 100))
console.log(dists.exponential_2p.measurements.stats.skewness(0.1, 100))
console.log(dists.exponential_2p.measurements.stats.kurtosis(0.1, 100))
// console.log(dists.exponential_2p.measurements.stats.median(0.1, 100))
console.log(dists.exponential_2p.measurements.stats.mode(0.1, 100))

// mean_value: 110
// variance_value: 99.99999999999999
// standard_deviation_value: 10
// skewness_value: 2
// kurtosis_value: 9
// median_value: 106.93147180559946
// mode_value: 100