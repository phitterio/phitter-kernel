jStat = require("../node_modules/jstat");

dists = {
    levy: {
        measurements: {
            nonCentralMoments: function (k, mu, c) {
                return undefined;
            },
            centralMoments: function (k, mu, c) {
                return undefined;
            },
            stats: {
                mean: function (mu, c) {
                    return Infinity;
                },
                variance: function (mu, c) {
                    return Infinity;
                },
                standardDeviation: function (mu, c) {
                    return Math.sqrt(this.variance(mu, c));
                },
                skewness: function (mu, c) {
                    return undefined;
                },
                kurtosis: function (mu, c) {
                    return undefined;
                },
                median: function (mu, c) {
                    return dists.levy.measurements.ppf(0.5, mu, c);
                },
                mode: function (mu, c) {
                    return mu + c / 3;
                },
            },
        }
    }
}
console.log(dists.levy.measurements.stats.mean(10, 2))
console.log(dists.levy.measurements.stats.variance(10, 2))
console.log(dists.levy.measurements.stats.standardDeviation(10, 2))
console.log(dists.levy.measurements.stats.skewness(10, 2))
console.log(dists.levy.measurements.stats.kurtosis(10, 2))
// console.log(dists.levy.measurements.stats.median(10, 2))
console.log(dists.levy.measurements.stats.mode(10, 2))

// mean_value: inf
// variance_value: inf
// standard_deviation_value: inf
// skewness_value: undefined
// kurtosis_value: undefined
// median_value: 14.396218676635463
// mode_value: 10.666666666666666