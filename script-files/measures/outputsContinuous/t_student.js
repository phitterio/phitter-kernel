jStat = require("../node_modules/jstat");

dists = {
    t_student: {
        measurements: {
            nonCentralMoments: function (k, df) {
                return undefined;
            },
            centralMoments: function (k, df) {
                return undefined;
            },
            stats: {
                mean: function (df) {
                    return 0;
                },
                variance: function (df) {
                    return df / (df - 2);
                },
                standardDeviation: function (df) {
                    return Math.sqrt(this.variance(df));
                },
                skewness: function (df) {
                    return 0;
                },
                kurtosis: function (df) {
                    return 6 / (df - 4) + 3;
                },
                median: function (df) {
                    return dists.t_student.measurements.ppf(0.5, df);
                },
                mode: function (df) {
                    return 0;
                },
            },
        }
    }
}
console.log(dists.t_student.measurements.stats.mean(10))
console.log(dists.t_student.measurements.stats.variance(10))
console.log(dists.t_student.measurements.stats.standardDeviation(10))
console.log(dists.t_student.measurements.stats.skewness(10))
console.log(dists.t_student.measurements.stats.kurtosis(10))
// console.log(dists.t_student.measurements.stats.median(10))
console.log(dists.t_student.measurements.stats.mode(10))

// mean_value: 0
// variance_value: 1.25
// standard_deviation_value: 1.0772477334018427
// skewness_value: 0
// kurtosis_value: 4
// median_value: 0
// mode_value: 0