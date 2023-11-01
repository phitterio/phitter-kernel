jStat = require("../node_modules/jstat");

dists = {
    moyal: {
        measurements: {
            nonCentralMoments: function (k, mu, sigma) {
                return undefined;
            },
            centralMoments: function (k, mu, sigma) {
                return undefined;
            },
            stats: {
                mean: function (mu, sigma) {
                    return mu + sigma * (Math.log(2) + 0.577215664901532);
                },
                variance: function (mu, sigma) {
                    return sigma * sigma * Math.PI * Math.PI / 2;
                },
                standardDeviation: function (mu, sigma) {
                    return Math.sqrt(this.variance(mu, sigma));
                },
                skewness: function (mu, sigma) {
                    return 1.5351415907229;
                },
                kurtosis: function (mu, sigma) {
                    return 7;
                },
                median: function (mu, sigma) {
                    return dists.moyal.measurements.ppf(0.5, mu, sigma);
                },
                mode: function (mu, sigma) {
                    return mu;
                },
            },
        }
    }
}
console.log(dists.moyal.measurements.stats.mean(20, 9))
console.log(dists.moyal.measurements.stats.variance(20, 9))
console.log(dists.moyal.measurements.stats.standardDeviation(20, 9))
console.log(dists.moyal.measurements.stats.skewness(20, 9))
console.log(dists.moyal.measurements.stats.kurtosis(20, 9))
// console.log(dists.moyal.measurements.stats.median(20, 9))
console.log(dists.moyal.measurements.stats.mode(20, 9))

// mean_value: 31.433265609153295
// variance_value: 399.718978244119
// standard_deviation_value: 19.99297322171265
// skewness_value: 1.5351415907229
// kurtosis_value: 7
// median_value: 27.088378392816033
// mode_value: 20