jStat = require("../node_modules/jstat");

dists = {
    generalized_normal: {
        measurements: {
            nonCentralMoments: function (k, miu, alpha, beta) {
                return undefined;
            },
            centralMoments: function (k, miu, alpha, beta) {
                return undefined;
            },
            stats: {
                mean: function (miu, alpha, beta) {
                    return beta;
                },
                variance: function (miu, alpha, beta) {
                    return ((miu) ** 2) * jStat.gammafn(3 / alpha) / jStat.gammafn(1 / alpha);
                },
                standardDeviation: function (miu, alpha, beta) {
                    return Math.sqrt(this.variance(miu, alpha, beta));
                },
                skewness: function (miu, alpha, beta) {
                    return 0;
                },
                kurtosis: function (miu, alpha, beta) {
                    return jStat.gammafn(5 / alpha) * jStat.gammafn(1 / alpha) / (jStat.gammafn(3 / alpha) ** 2);
                },
                median: function (miu, alpha, beta) {
                    return dists.generalized_normal.measurements.ppf(0.5, miu, alpha, beta);
                },
                mode: function (miu, alpha, beta) {
                    return beta;
                },
            },
        }
    }
}
console.log(dists.generalized_normal.measurements.stats.mean(3, 4, 20))
console.log(dists.generalized_normal.measurements.stats.variance(3, 4, 20))
console.log(dists.generalized_normal.measurements.stats.standardDeviation(3, 4, 20))
console.log(dists.generalized_normal.measurements.stats.skewness(3, 4, 20))
console.log(dists.generalized_normal.measurements.stats.kurtosis(3, 4, 20))
// console.log(dists.generalized_normal.measurements.stats.median(3, 4, 20))
console.log(dists.generalized_normal.measurements.stats.mode(3, 4, 20))

// mean_value: 20
// variance_value: 3.041902080302781
// standard_deviation_value: 1.7441049510573556
// skewness_value: 0
// kurtosis_value: 2.1884396152264767
// median_value: 20
// mode_value: 20