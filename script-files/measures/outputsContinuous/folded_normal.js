jStat = require("../node_modules/jstat");

dists = {
    folded_normal: {
        measurements: {
            nonCentralMoments: function (k, miu, sigma) {
                return undefined;
            },
            centralMoments: function (k, miu, sigma) {
                return undefined;
            },
            stats: {
                mean: function (miu, sigma) {
                    const std_cdf = (t) => 0.5 * (1 + jStat.erf(t / Math.sqrt(2)));
                    return sigma * Math.sqrt(2 / Math.PI) * Math.exp(-miu * miu / (2 * sigma * sigma)) - miu * (2 * std_cdf(-miu / sigma) - 1);
                },
                variance: function (miu, sigma) {
                    return miu * miu + sigma * sigma - this.mean(miu, sigma) ** 2;
                },
                standardDeviation: function (miu, sigma) {
                    return Math.sqrt(this.variance(miu, sigma));
                },
                skewness: function (miu, sigma) {
                    return undefined;
                },
                kurtosis: function (miu, sigma) {
                    return undefined;
                },
                median: function (miu, sigma) {
                    return dists.folded_normal.measurements.ppf(0.5, miu, sigma);
                },
                mode: function (miu, sigma) {
                    return miu;
                },
            },
        }
    }
}
console.log(dists.folded_normal.measurements.stats.mean(1500, 600))
console.log(dists.folded_normal.measurements.stats.variance(1500, 600))
console.log(dists.folded_normal.measurements.stats.standardDeviation(1500, 600))
console.log(dists.folded_normal.measurements.stats.skewness(1500, 600))
console.log(dists.folded_normal.measurements.stats.kurtosis(1500, 600))
// console.log(dists.folded_normal.measurements.stats.median(1500, 600))
console.log(dists.folded_normal.measurements.stats.mode(1500, 600))

// mean_value: 1502.4049646149538
// variance_value: 352779.3223003396
// standard_deviation_value: 593.9522895825385
// skewness_value: None
// kurtosis_value: None
// median_value: 1500
// mode_value: 1500