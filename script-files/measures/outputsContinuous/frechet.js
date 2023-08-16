jStat = require("../node_modules/jstat");

dists = {
    frechet: {
        measurements: {
            nonCentralMoments: function (k, alpha, loc, scale) {
                return jStat.gammafn(1 - k / alpha)
            },
            centralMoments: function (k, alpha, loc, scale) {
                const µ1 = this.nonCentralMoments(1, alpha, loc, scale);
                const µ2 = this.nonCentralMoments(2, alpha, loc, scale);
                const µ3 = this.nonCentralMoments(3, alpha, loc, scale);
                const µ4 = this.nonCentralMoments(4, alpha, loc, scale);

                let result;
                switch (k) {
                    case 1: result = 0; break;
                    case 2: result = µ2 - µ1 ** 2; break;
                    case 3: result = µ3 - 3 * µ1 * µ2 + 2 * µ1 ** 3; break;
                    case 4: result = µ4 - 4 * µ1 * µ3 + 6 * (µ1 ** 2) * µ2 - 3 * (µ1 ** 4); break;
                };
                return result
            },
            stats: {
                mean: function (alpha, loc, scale) {
                    const µ1 = dists.frechet.measurements.nonCentralMoments(1, alpha, loc, scale);
                    return loc + scale * µ1;
                },
                variance: function (alpha, loc, scale) {
                    const µ1 = dists.frechet.measurements.nonCentralMoments(1, alpha, loc, scale);
                    const µ2 = dists.frechet.measurements.nonCentralMoments(2, alpha, loc, scale);
                    return (scale ** 2) * (µ2 - µ1 ** 2);
                },
                standardDeviation: function (alpha, loc, scale) {
                    return Math.sqrt(this.variance(alpha, loc, scale));
                },
                skewness: function (alpha, loc, scale) {
                    const central_µ3 = dists.frechet.measurements.centralMoments(3, alpha, loc, scale);
                    const µ1 = dists.frechet.measurements.nonCentralMoments(1, alpha, loc, scale);
                    const µ2 = dists.frechet.measurements.nonCentralMoments(2, alpha, loc, scale);
                    const std = Math.sqrt(µ2 - µ1 ** 2);
                    return central_µ3 / (std ** 3);
                },
                kurtosis: function (alpha, loc, scale) {
                    const central_µ4 = dists.frechet.measurements.centralMoments(4, alpha, loc, scale);
                    const µ1 = dists.frechet.measurements.nonCentralMoments(1, alpha, loc, scale);
                    const µ2 = dists.frechet.measurements.nonCentralMoments(2, alpha, loc, scale);
                    const std = Math.sqrt(µ2 - µ1 ** 2);
                    return central_µ4 / (std ** 4);
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