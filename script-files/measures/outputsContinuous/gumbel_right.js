jStat = require("../node_modules/jstat");

dists = {
    gumbel_right: {
        measurements: {
            nonCentralMoments: function (k, miu, sigma) {
                return undefined;
            },
            centralMoments: function (k, miu, sigma) {
                return undefined;
            },
            stats: {
                mean: function (miu, sigma) {
                    return miu+0.5772156649*sigma;
                },
                variance: function (miu, sigma) {
                    return (sigma**2)*((Math.PI**2)/6);
                },
                standardDeviation: function (miu, sigma) {
                    return Math.sqrt(this.variance(miu, sigma));
                },
                skewness: function (miu, sigma) {
                    return 12*Math.sqrt(6)*1.20205690315959/(Math.PI**3);
                },
                kurtosis: function (miu, sigma) {
                    return 3+(12/5);
                },
                median: function (miu, sigma) {
                    return dists.gumbel_right.measurements.ppf(0.5, miu, sigma);
                },
                mode: function (miu, sigma) {
                    return miu;
                },
            },
        }
    }
}
console.log(dists.gumbel_right.measurements.stats.mean(1000, 159))
console.log(dists.gumbel_right.measurements.stats.variance(1000, 159))
console.log(dists.gumbel_right.measurements.stats.standardDeviation(1000, 159))
console.log(dists.gumbel_right.measurements.stats.skewness(1000, 159))
console.log(dists.gumbel_right.measurements.stats.kurtosis(1000, 159))
// console.log(dists.gumbel_right.measurements.stats.median(1000, 159))
console.log(dists.gumbel_right.measurements.stats.mode(1000, 159))

// mean_value: 1091.7772907191
// variance_value: 41585.57814399001
// standard_deviation_value: 203.92542299573637
// skewness_value: 1.1395470994046446
// kurtosis_value: 5.4
// median_value: 1058.2755543724847
// mode_value: 1000