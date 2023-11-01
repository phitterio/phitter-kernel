jStat = require("../node_modules/jstat");

dists = {
    frechet: {
        measurements: {
            nonCentralMoments: function (k, alpha, loc, scale) {
                return jStat.gammafn(1 - k / alpha)
            },
            centralMoments: function (k, alpha, loc, scale) {
                const miu1 = this.nonCentralMoments(1, alpha, loc, scale);
                const miu2 = this.nonCentralMoments(2, alpha, loc, scale);
                const miu3 = this.nonCentralMoments(3, alpha, loc, scale);
                const miu4 = this.nonCentralMoments(4, alpha, loc, scale);

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
                mean: function (alpha, loc, scale) {
                    const miu1 = dists.frechet.measurements.nonCentralMoments(1, alpha, loc, scale);
                    return loc + scale * miu1;
                },
                variance: function (alpha, loc, scale) {
                    const miu1 = dists.frechet.measurements.nonCentralMoments(1, alpha, loc, scale);
                    const miu2 = dists.frechet.measurements.nonCentralMoments(2, alpha, loc, scale);
                    return (scale ** 2) * (miu2 - miu1 ** 2);
                },
                standardDeviation: function (alpha, loc, scale) {
                    return Math.sqrt(this.variance(alpha, loc, scale));
                },
                skewness: function (alpha, loc, scale) {
                    const central_miu3 = dists.frechet.measurements.centralMoments(3, alpha, loc, scale);
                    const miu1 = dists.frechet.measurements.nonCentralMoments(1, alpha, loc, scale);
                    const miu2 = dists.frechet.measurements.nonCentralMoments(2, alpha, loc, scale);
                    const std = Math.sqrt(miu2 - miu1 ** 2);
                    return central_miu3 / (std ** 3);
                },
                kurtosis: function (alpha, loc, scale) {
                    const central_miu4 = dists.frechet.measurements.centralMoments(4, alpha, loc, scale);
                    const miu1 = dists.frechet.measurements.nonCentralMoments(1, alpha, loc, scale);
                    const miu2 = dists.frechet.measurements.nonCentralMoments(2, alpha, loc, scale);
                    const std = Math.sqrt(miu2 - miu1 ** 2);
                    return central_miu4 / (std ** 4);
                },
                median: function (alpha, loc, scale) {
                    return dists.frechet.measurements.ppf(0.5, alpha, loc, scale);
                },
                mode: function (alpha, loc, scale) {
                    return loc + scale * (alpha / (alpha + 1)) ** (1 / alpha);
                },
            },
        }
    }
}
console.log(dists.frechet.measurements.stats.mean(5, 10, 20))
console.log(dists.frechet.measurements.stats.variance(5, 10, 20))
console.log(dists.frechet.measurements.stats.standardDeviation(5, 10, 20))
console.log(dists.frechet.measurements.stats.skewness(5, 10, 20))
console.log(dists.frechet.measurements.stats.kurtosis(5, 10, 20))
// console.log(dists.frechet.measurements.stats.median(5, 10, 20))
console.log(dists.frechet.measurements.stats.mode(5, 10, 20))

// mean_value: 33.28459427450608
// variance_value: 53.50456899676574
// standard_deviation_value: 7.314681742684759
// skewness_value: 3.5350716046214425
// kurtosis_value: 48.091512125816415
// median_value: 31.5211217027801
// mode_value: 29.283850080052545