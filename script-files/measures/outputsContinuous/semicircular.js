jStat = require("../node_modules/jstat");

dists = {
    semicircular: {
        measurements: {
            nonCentralMoments: function (k, loc, R) {
                return undefined;
            },
            centralMoments: function (k, loc, R) {
                return undefined;
            },
            stats: {
                mean: function (loc, R) {
                    return loc;
                },
                variance: function (loc, R) {
                    return R * R / 4;
                },
                standardDeviation: function (loc, R) {
                    return Math.sqrt(this.variance(loc, R));
                },
                skewness: function (loc, R) {
                    return 0;
                },
                kurtosis: function (loc, R) {
                    return 2;
                },
                median: function (loc, R) {
                    return dists.semicircular.measurements.ppf(0.5, loc, R);
                },
                mode: function (loc, R) {
                    return loc;
                },
            },
        }
    }
}
console.log(dists.semicircular.measurements.stats.mean(20, 5))
console.log(dists.semicircular.measurements.stats.variance(20, 5))
console.log(dists.semicircular.measurements.stats.standardDeviation(20, 5))
console.log(dists.semicircular.measurements.stats.skewness(20, 5))
console.log(dists.semicircular.measurements.stats.kurtosis(20, 5))
// console.log(dists.semicircular.measurements.stats.median(20, 5))
console.log(dists.semicircular.measurements.stats.mode(20, 5))

// mean_value: 20
// variance_value: 6.25
// standard_deviation_value: 2.5
// skewness_value: 0
// kurtosis_value: 2
// median_value: 20
// mode_value: 20