jStat = require("../node_modules/jstat");

dists = {
    f_4p: {
        measurements: {
            nonCentralMoments: function (k, df1, df2, loc, scale) {
                return ((df2 / df1) ** k) * (jStat.gammafn(df1 / 2 + k) / jStat.gammafn(df1 / 2)) * (jStat.gammafn(df2 / 2 - k) / jStat.gammafn(df2 / 2))
            },
            centralMoments: function (k, df1, df2, loc, scale) {
                const µ1 = this.nonCentralMoments(1, df1, df2, loc, scale);
                const µ2 = this.nonCentralMoments(2, df1, df2, loc, scale);
                const µ3 = this.nonCentralMoments(3, df1, df2, loc, scale);
                const µ4 = this.nonCentralMoments(4, df1, df2, loc, scale);

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
                mean: function (df1, df2, loc, scale) {
                    const µ1 = dists.f_4p.measurements.nonCentralMoments(1, df1, df2, loc, scale);
                    return loc + scale * µ1;
                },
                variance: function (df1, df2, loc, scale) {
                    const µ1 = dists.f_4p.measurements.nonCentralMoments(1, df1, df2, loc, scale);
                    const µ2 = dists.f_4p.measurements.nonCentralMoments(2, df1, df2, loc, scale);
                    return (scale ** 2) * (µ2 - µ1 ** 2);
                },
                standardDeviation: function (df1, df2, loc, scale) {
                    return Math.sqrt(this.variance(df1, df2, loc, scale));
                },
                skewness: function (df1, df2, loc, scale) {
                    const central_µ3 = dists.f_4p.measurements.centralMoments(3, df1, df2, loc, scale);
                    return central_µ3 / (this.standardDeviation(df1, df2, loc, scale) ** 3);
                },
                kurtosis: function (df1, df2, loc, scale) {
                    const central_µ4 = dists.f_4p.measurements.centralMoments(4, df1, df2, loc, scale);
                    return central_µ4 / (this.standardDeviation(df1, df2, loc, scale) ** 4);
                },
                median: function (df1, df2, loc, scale) {
                    return dists.f_4p.measurements.ppf(0.5, df1, df2, loc, scale);
                },
                mode: function (df1, df2, loc, scale) {
                    return (df2 * (df1 - 2)) / (df1 * (df2 + 2)) * scale + loc;
                },
            },
        }
    }
}
console.log(dists.f_4p.measurements.stats.mean(15, 9, 100, 70))
console.log(dists.f_4p.measurements.stats.variance(15, 9, 100, 70))
console.log(dists.f_4p.measurements.stats.standardDeviation(15, 9, 100, 70))
console.log(dists.f_4p.measurements.stats.skewness(15, 9, 100, 70))
console.log(dists.f_4p.measurements.stats.kurtosis(15, 9, 100, 70))
// console.log(dists.f_4p.measurements.stats.median(15, 9, 100, 70))
console.log(dists.f_4p.measurements.stats.mode(15, 9, 100, 70))

// mean_value: 190
// variance_value: 4752.000000000002
// standard_deviation_value: 68.93475175845636
// skewness_value: 4.293915513573877
// kurtosis_value: 97.96969696969691
// median_value: 172.17927737861805
// mode_value: 149.63636363636363