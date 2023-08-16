jStat = require("../node_modules/jstat");

dists = {
    chi_square_3p: {
        measurements: {
            nonCentralMoments: function (k, df, loc, scale) {
                return undefined;
            },
            centralMoments: function (k, df, loc, scale) {
                return undefined;
            },
            stats: {
                mean: function (df, loc, scale) {
                    return df * scale + loc;
                },
                variance: function (df, loc, scale) {
                    return (df * 2) * (scale * scale);
                },
                standardDeviation: function (df, loc, scale) {
                    return Math.sqrt(this.variance(df, loc, scale));
                },
                skewness: function (df, loc, scale) {
                    return Math.sqrt(8 / df);
                },
                kurtosis: function (df, loc, scale) {
                    return 12 / df + 3;
                },
                median: function (df, loc, scale) {
                    return dists.chi_square_3p.measurements.ppf(0.5, df, loc, scale);
                },
                mode: function (df, loc, scale) {
                    return (df - 2) * scale + loc;
                },
            },
        }
    }
}
console.log(dists.chi_square_3p.measurements.stats.mean(5, 10, 2))
console.log(dists.chi_square_3p.measurements.stats.variance(5, 10, 2))
console.log(dists.chi_square_3p.measurements.stats.standardDeviation(5, 10, 2))
console.log(dists.chi_square_3p.measurements.stats.skewness(5, 10, 2))
console.log(dists.chi_square_3p.measurements.stats.kurtosis(5, 10, 2))
// console.log(dists.chi_square_3p.measurements.stats.median(5, 10, 2))
console.log(dists.chi_square_3p.measurements.stats.mode(5, 10, 2))

// mean_value: 20
// variance_value: 40
// standard_deviation_value: 6.324555320336759
// skewness_value: 1.2649110640673518
// kurtosis_value: 5.4
// median_value: 18.725048010973936
// mode_value: 16