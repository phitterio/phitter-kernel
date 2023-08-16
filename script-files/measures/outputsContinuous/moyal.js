jStat = require("../node_modules/jstat");

dists = {
    moyal: {
        measurements: {
            nonCentralMoments: function (k, miu, sigma) {
                return undefined;
            },
            centralMoments: function (k, miu, sigma) {
                return undefined;
            },
            stats: {
                mean: function (miu, sigma) {
                    return miu + sigma * (Math.log(2) + 0.577215664901532);
                },
                variance: function (miu, sigma) {
                    return sigma * sigma * Math.PI * Math.PI / 2;
                },
                standardDeviation: function (miu, sigma) {
                    return Math.sqrt(this.variance(miu, sigma));
                },
                skewness: function (miu, sigma) {
                    return 1.5351415907229;
                },
                kurtosis: function (miu, sigma) {
                    return 7;
                },
                median: function (miu, sigma) {
                    return dists.moyal.measurements.ppf(0.5, miu, sigma);
                },
                mode: function (miu, sigma) {
                    return miu;
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