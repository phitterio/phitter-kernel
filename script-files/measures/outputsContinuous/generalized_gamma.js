jStat = require("../node_modules/jstat");

dists = {
    generalized_gamma: {
        measurements: {
            nonCentralMoments: function (k, a, d, p) {
                return (a ** k) * jStat.gammafn((d + k) / p) / jStat.gammafn((d) / p)
            },
            centralMoments: function (k, a, d, p) {
                const miu1 = this.nonCentralMoments(1, a, d, p);
                const miu2 = this.nonCentralMoments(2, a, d, p);
                const miu3 = this.nonCentralMoments(3, a, d, p);
                const miu4 = this.nonCentralMoments(4, a, d, p);

                let result;
                switch (k) {
                    case 1: result = 0; break;
                    case 2: result = miu2 - miu1 ** 2; break;
                    case 3: result = miu3 - 3 * miu1 * miu2 + 2 * miu1 ** 3; break;
                    case 4: result = miu4 - 4 * miu1 * miu3 + 6 * (miu1 ** 2) * miu2 - 3 * (miu1 ** 4); break;
                };
                return result
            },
            stats: {
                mean: function (a, d, p) {
                    const miu1 = dists.generalized_gamma.measurements.nonCentralMoments(1, a, d, p);
                    return miu1;
                },
                variance: function (a, d, p) {
                    const miu1 = dists.generalized_gamma.measurements.nonCentralMoments(1, a, d, p);
                    const miu2 = dists.generalized_gamma.measurements.nonCentralMoments(2, a, d, p);
                    return miu2 - miu1 ** 2;
                },
                standardDeviation: function (a, d, p) {
                    return Math.sqrt(this.variance(a, d, p));
                },
                skewness: function (a, d, p) {
                    const central_miu3 = dists.generalized_gamma.measurements.centralMoments(3, a, d, p);
                    return central_miu3 / (this.standardDeviation(a, d, p) ** 3);
                },
                kurtosis: function (a, d, p) {
                    const central_miu4 = dists.generalized_gamma.measurements.centralMoments(4, a, d, p);
                    return central_miu4 / (this.standardDeviation(a, d, p) ** 4);
                },
                median: function (a, d, p) {
                    return dists.generalized_gamma.measurements.ppf(0.5, a, d, p);
                },
                mode: function (a, d, p) {
                    return a * ((d - 1) / p) ** (1 / p);
                },
            },
        }
    }
}
console.log(dists.generalized_gamma.measurements.stats.mean(10, 150, 20))
console.log(dists.generalized_gamma.measurements.stats.variance(10, 150, 20))
console.log(dists.generalized_gamma.measurements.stats.standardDeviation(10, 150, 20))
console.log(dists.generalized_gamma.measurements.stats.skewness(10, 150, 20))
console.log(dists.generalized_gamma.measurements.stats.kurtosis(10, 150, 20))
// console.log(dists.generalized_gamma.measurements.stats.median(10, 150, 20))
console.log(dists.generalized_gamma.measurements.stats.mode(10, 150, 20))

// mean_value: 11.024287744229486
// variance_value: 0.04303370606395163
// standard_deviation_value: 0.20744567014992535
// skewness_value: -0.4409436687911422
// kurtosis_value: 3.1980826667115974
// median_value: 11.03504810450817
// mode_value: 11.056249067821739