jStat = require("../node_modules/jstat");

dists = {
    normal: {
        measurements: {
            nonCentralMoments: function (k, miu, sigma) {
                return undefined;
            },
            centralMoments: function (k, miu, sigma) {
                return undefined;
            },
            stats: {
                mean: function (miu, sigma) {
                    return miu;
                },
                variance: function (miu, sigma) {
                    return sigma * 3;
                },
                standardDeviation: function (miu, sigma) {
                    return Math.sqrt(this.variance(miu, sigma));
                },
                skewness: function (miu, sigma) {
                    return 0;
                },
                kurtosis: function (miu, sigma) {
                    return 3;
                },
                median: function (miu, sigma) {
                    return dists.normal.measurements.ppf(0.5, miu, sigma);
                },
                mode: function (miu, sigma) {
                    return miu;
                },
            },
        }
    }
}
console.log(dists.normal.measurements.stats.mean(5, 3))
console.log(dists.normal.measurements.stats.variance(5, 3))
console.log(dists.normal.measurements.stats.standardDeviation(5, 3))
console.log(dists.normal.measurements.stats.skewness(5, 3))
console.log(dists.normal.measurements.stats.kurtosis(5, 3))
// console.log(dists.normal.measurements.stats.median(5, 3))
console.log(dists.normal.measurements.stats.mode(5, 3))

// mean_value: 5
// variance_value: 9
// standard_deviation_value: 3
// skewness_value: 0
// kurtosis_value: 3
// median_value: 5
// mode_value: 5