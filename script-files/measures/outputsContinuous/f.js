jStat = require("../node_modules/jstat");

dists = {
    f: {
        measurements: {
            nonCentralMoments: function (k, df1, df2) {
                return ((df2 / df1) ** k) * (jStat.gammafn(df1 / 2 + k) / jStat.gammafn(df1 / 2)) * (jStat.gammafn(df2 / 2 - k) / jStat.gammafn(df2 / 2))
            },
            centralMoments: function (k, df1, df2) {
                const µ1 = this.nonCentralMoments(1, df1, df2);
                const µ2 = this.nonCentralMoments(2, df1, df2);
                const µ3 = this.nonCentralMoments(3, df1, df2);
                const µ4 = this.nonCentralMoments(4, df1, df2);

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
                mean: function (df1, df2) {
                    const µ1 = dists.f.measurements.nonCentralMoments(1, df1, df2);
                    return µ1;
                },
                variance: function (df1, df2) {
                    const µ1 = dists.f.measurements.nonCentralMoments(1, df1, df2);
                    const µ2 = dists.f.measurements.nonCentralMoments(2, df1, df2);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (df1, df2) {
                    return Math.sqrt(this.variance(df1, df2));
                },
                skewness: function (df1, df2) {
                    const central_µ3 = dists.f.measurements.centralMoments(3, df1, df2);
                    return central_µ3 / (this.standardDeviation(df1, df2) ** 3);
                },
                kurtosis: function (df1, df2) {
                    const central_µ4 = dists.f.measurements.centralMoments(4, df1, df2);
                    return central_µ4 / (this.standardDeviation(df1, df2) ** 4);
                },
                median: function (df1, df2) {
                    return dists.f.measurements.ppf(0.5, df1, df2);
                },
                mode: function (df1, df2) {
                    return (df2 * (df1 - 2)) / (df1 * (df2 + 2));
                },
            },
        }
    }
}
console.log(dists.f.measurements.stats.mean(15, 20))
console.log(dists.f.measurements.stats.variance(15, 20))
console.log(dists.f.measurements.stats.standardDeviation(15, 20))
console.log(dists.f.measurements.stats.skewness(15, 20))
console.log(dists.f.measurements.stats.kurtosis(15, 20))
// console.log(dists.f.measurements.stats.median(15, 20))
console.log(dists.f.measurements.stats.mode(15, 20))

// mean_value: 1.1111111111111112
// variance_value: 0.3395061728395059
// standard_deviation_value: 0.5826715823167506
// skewness_value: 1.7434744489062268
// kurtosis_value: 9.319480519480532
// median_value: 0.9886983373954129
// mode_value: 0.7878787878787878