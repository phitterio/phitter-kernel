jStat = require("../node_modules/jstat");

dists = {
    levy: {
        measurements: {
            nonCentralMoments: function (k, miu, c) {
                return undefined;
            },
            centralMoments: function (k, miu, c) {
                return undefined;
            },
            stats: {
                mean: function (miu, c) {
                    return Infinity;
                },
                variance: function (miu, c) {
                    return Infinity;
                },
                standardDeviation: function (miu, c) {
                    return Math.sqrt(this.variance(miu, c));
                },
                skewness: function (miu, c) {
                    return undefined;
                },
                kurtosis: function (miu, c) {
                    return undefined;
                },
                median: function (miu, c) {
                    return dists.levy.measurements.ppf(0.5, miu, c);
                },
                mode: function (miu, c) {
                    return miu + c / 3;
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