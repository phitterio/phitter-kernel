jStat = require("../node_modules/jstat");

dists = {
    t_student_3p: {
        measurements: {
            nonCentralMoments: function (k, df, loc, scale) {
                return undefined;
            },
            centralMoments: function (k, df, loc, scale) {
                return undefined;
            },
            stats: {
                mean: function (df, loc, scale) {
                    return loc;
                },
                variance: function (df, loc, scale) {
                    return scale * scale * df / (df - 2);
                },
                standardDeviation: function (df, loc, scale) {
                    return Math.sqrt(this.variance(df, loc, scale));
                },
                skewness: function (df, loc, scale) {
                    return 0;
                },
                kurtosis: function (df, loc, scale) {
                    return 6 / (df - 4) + 3;
                },
                median: function (df, loc, scale) {
                    return dists.t_student_3p.measurements.ppf(0.5, df, loc, scale);
                },
                mode: function (df, loc, scale) {
                    return loc;
                },
            },
        }
    }
}
console.log(dists.t_student_3p.measurements.stats.mean(24, 100, 3))
console.log(dists.t_student_3p.measurements.stats.variance(24, 100, 3))
console.log(dists.t_student_3p.measurements.stats.standardDeviation(24, 100, 3))
console.log(dists.t_student_3p.measurements.stats.skewness(24, 100, 3))
console.log(dists.t_student_3p.measurements.stats.kurtosis(24, 100, 3))
// console.log(dists.t_student_3p.measurements.stats.median(24, 100, 3))
console.log(dists.t_student_3p.measurements.stats.mode(24, 100, 3))

// mean_value: 100
// variance_value: 9.818181818181818
// standard_deviation_value: 3.1333978072025612
// skewness_value: 0
// kurtosis_value: 3.3
// median_value: 100
// mode_value: 100