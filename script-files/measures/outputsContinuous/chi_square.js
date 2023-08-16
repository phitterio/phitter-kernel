jStat = require("../node_modules/jstat");

dists = {
    chi_square: {
        measurements: {
            nonCentralMoments: function (k, df) {
                return undefined;
            },
            centralMoments: function (k, df) {
                return undefined;
            },
            stats: {
                mean: function (df) {
                    return df;
                },
                variance: function (df) {
                    return df * 2;
                },
                standardDeviation: function (df) {
                    return Math.sqrt(this.variance(df));
                },
                skewness: function (df) {
                    return Math.sqrt(8 / df);
                },
                kurtosis: function (df) {
                    return 12 / df + 3;
                },
                median: function (df) {
                    return dists.chi_square.measurements.ppf(0.5, df);
                },
                mode: function (df) {
                    return df - 2;
                },
            },
        }
    }
}
console.log(dists.chi_square.measurements.stats.mean(7))
console.log(dists.chi_square.measurements.stats.variance(7))
console.log(dists.chi_square.measurements.stats.standardDeviation(7))
console.log(dists.chi_square.measurements.stats.skewness(7))
console.log(dists.chi_square.measurements.stats.kurtosis(7))
// console.log(dists.chi_square.measurements.stats.median(7))
console.log(dists.chi_square.measurements.stats.mode(7))

// mean_value: 7
// variance_value: 14
// standard_deviation_value: 3.7416573867739413
// skewness_value: 1.0690449676496976
// kurtosis_value: 4.714285714285714
// median_value: 6.354273396601439
// mode_value: 5