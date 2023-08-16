jStat = require("../node_modules/jstat");

dists = {
    gumbel_left: {
        measurements: {
            nonCentralMoments: function (k, miu, sigma) {
                return undefined;
            },
            centralMoments: function (k, miu, sigma) {
                return undefined;
            },
            stats: {
                mean: function (miu, sigma) {
                    return miu-0.5772156649*sigma;
                },
                variance: function (miu, sigma) {
                    return (sigma**2)*((Math.PI**2)/6);
                },
                standardDeviation: function (miu, sigma) {
                    return Math.sqrt(this.variance(miu, sigma));
                },
                skewness: function (miu, sigma) {
                    return -12*Math.sqrt(6)*1.20205690315959/(Math.PI**3);
                },
                kurtosis: function (miu, sigma) {
                    return 3+(12/5);
                },
                median: function (miu, sigma) {
                    return dists.gumbel_left.measurements.ppf(0.5, miu, sigma);
                },
                mode: function (miu, sigma) {
                    return miu;
                },
            },
        }
    }
}
console.log(dists.gumbel_left.measurements.stats.mean(100, 30))
console.log(dists.gumbel_left.measurements.stats.variance(100, 30))
console.log(dists.gumbel_left.measurements.stats.standardDeviation(100, 30))
console.log(dists.gumbel_left.measurements.stats.skewness(100, 30))
console.log(dists.gumbel_left.measurements.stats.kurtosis(100, 30))
// console.log(dists.gumbel_left.measurements.stats.median(100, 30))
console.log(dists.gumbel_left.measurements.stats.mode(100, 30))

// mean_value: 82.683530053
// variance_value: 1480.4406601634037
// standard_deviation_value: 38.47649490485592
// skewness_value: -1.1395470994046446
// kurtosis_value: 5.4
// median_value: 89.00461238255006
// mode_value: 100