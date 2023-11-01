jStat = require("../node_modules/jstat");

dists = {
    half_normal: {
        measurements: {
            nonCentralMoments: function (k, mu, sigma) {
                return undefined;
            },
            centralMoments: function (k, mu, sigma) {
                return undefined;
            },
            stats: {
                mean: function (mu, sigma) {
                    return mu + sigma * Math.sqrt(2 / Math.PI);
                },
                variance: function (mu, sigma) {
                    return sigma * sigma * (1 - 2 / Math.PI);
                },
                standardDeviation: function (mu, sigma) {
                    return Math.sqrt(this.variance(mu, sigma));
                },
                skewness: function (mu, sigma) {
                    return Math.sqrt(2) * (4 - Math.PI) / ((Math.PI - 2) ** 1.5);
                },
                kurtosis: function (mu, sigma) {
                    return 3 + 8 * (Math.PI - 3) / ((Math.PI - 2) ** 2);
                },
                median: function (mu, sigma) {
                    return dists.half_normal.measurements.ppf(0.5, mu, sigma);
                },
                mode: function (mu, sigma) {
                    return mu;
                },
            },
        }
    }
}
console.log(dists.half_normal.measurements.stats.mean(20, 7))
console.log(dists.half_normal.measurements.stats.variance(20, 7))
console.log(dists.half_normal.measurements.stats.standardDeviation(20, 7))
console.log(dists.half_normal.measurements.stats.skewness(20, 7))
console.log(dists.half_normal.measurements.stats.kurtosis(20, 7))
// console.log(dists.half_normal.measurements.stats.median(20, 7))
console.log(dists.half_normal.measurements.stats.mode(20, 7))

// mean_value: 25.58519192562006
// variance_value: 17.80563115398851
// standard_deviation_value: 4.219671924923609
// skewness_value: 0.9952717464311565
// kurtosis_value: 3.869177303605974
// median_value: 24.721428251372572
// mode_value: 20