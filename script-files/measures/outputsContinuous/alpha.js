jStat = require("jstat");

dists = {
    alpha: {
        measurements: {
            nonCentralMoments: function (k, alpha, loc, scale) {
                return undefined;
            },
            centralMoments: function (k, alpha, loc, scale) {
                return undefined;
            },
            stats: {
                mean: function (alpha, loc, scale) {
                    return undefined;
                },
                variance: function (alpha, loc, scale) {
                    return undefined;
                },
                standardDeviation: function (alpha, loc, scale) {
                    return Math.sqrt(this.variance(alpha, loc, scale));
                },
                skewness: function (alpha, loc, scale) {
                    return undefined;
                },
                kurtosis: function (alpha, loc, scale) {
                    return undefined;
                },
                median: function (alpha, loc, scale) {
                    return dists.alpha.measurements.ppf(0.5, alpha, loc, scale);
                },
                mode: function (alpha, loc, scale) {
                    return scale*(Math.sqrt(alpha*alpha+8)-alpha)/4+loc;
                },
            },
        }
    }
}
console.log(dists.alpha.measurements.stats.mean(5, 10, 49))
console.log(dists.alpha.measurements.stats.variance(5, 10, 49))
console.log(dists.alpha.measurements.stats.standardDeviation(5, 10, 49))
console.log(dists.alpha.measurements.stats.skewness(5, 10, 49))
console.log(dists.alpha.measurements.stats.kurtosis(5, 10, 49))
// console.log(dists.alpha.measurements.stats.median(5, 10, 49))
console.log(dists.alpha.measurements.stats.mode(5, 10, 49))

// mean_value: complicated
// variance_value: complicated
// standard_deviation_value: complicated
// skewness_value: complicated
// kurtosis_value: complicated
// median_value: 19.799999295841694
// mode_value: 19.12089242009085