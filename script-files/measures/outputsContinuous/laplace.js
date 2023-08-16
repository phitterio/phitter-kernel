jStat = require("../node_modules/jstat");

dists = {
    laplace: {
        measurements: {
            nonCentralMoments: function (k, miu, b) {
                return undefined;
            },
            centralMoments: function (k, miu, b) {
                return undefined;
            },
            stats: {
                mean: function (miu, b) {
                    return miu;
                },
                variance: function (miu, b) {
                    return 2 * b ** 2;
                },
                standardDeviation: function (miu, b) {
                    return Math.sqrt(this.variance(miu, b));
                },
                skewness: function (miu, b) {
                    return 0;
                },
                kurtosis: function (miu, b) {
                    return 6;
                },
                median: function (miu, b) {
                    return dists.laplace.measurements.ppf(0.5, miu, b);
                },
                mode: function (miu, b) {
                    return miu;
                },
            },
        }
    }
}
console.log(dists.laplace.measurements.stats.mean(17, 5))
console.log(dists.laplace.measurements.stats.variance(17, 5))
console.log(dists.laplace.measurements.stats.standardDeviation(17, 5))
console.log(dists.laplace.measurements.stats.skewness(17, 5))
console.log(dists.laplace.measurements.stats.kurtosis(17, 5))
// console.log(dists.laplace.measurements.stats.median(17, 5))
console.log(dists.laplace.measurements.stats.mode(17, 5))

// mean_value: 17
// variance_value: 50
// standard_deviation_value: 7.0710678118654755
// skewness_value: 0
// kurtosis_value: 6
// median_value: 17
// mode_value: 17