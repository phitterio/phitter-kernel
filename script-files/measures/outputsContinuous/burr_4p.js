jStat = require("../node_modules/jstat");

dists = {
    burr_4p: {
        measurements: {
            nonCentralMoments: function (k, A, B, C, loc) {
                return (A ** k) * (C) * (jStat.gammafn((B * C - k) / B) * jStat.gammafn((B + k) / B) / jStat.gammafn(((B * C - k) / B) + ((B + k) / B)))
            },
            centralMoments: function (k, A, B, C, loc) {
                const µ1 = this.nonCentralMoments(1, A, B, C, loc);
                const µ2 = this.nonCentralMoments(2, A, B, C, loc);
                const µ3 = this.nonCentralMoments(3, A, B, C, loc);
                const µ4 = this.nonCentralMoments(4, A, B, C, loc);

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
                mean: function (A, B, C, loc) {
                    const µ1 = dists.burr_4p.measurements.nonCentralMoments(1, A, B, C, loc);
                    return loc + µ1;
                },
                variance: function (A, B, C, loc) {
                    const µ1 = dists.burr_4p.measurements.nonCentralMoments(1, A, B, C, loc);
                    const µ2 = dists.burr_4p.measurements.nonCentralMoments(2, A, B, C, loc);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (A, B, C, loc) {
                    return Math.sqrt(this.variance(A, B, C, loc));
                },
                skewness: function (A, B, C, loc) {
                    const central_µ3 = dists.burr_4p.measurements.centralMoments(3, A, B, C, loc);
                    return central_µ3 / (this.standardDeviation(A, B, C, loc) ** 3);
                },
                kurtosis: function (A, B, C, loc) {
                    const central_µ4 = dists.burr_4p.measurements.centralMoments(4, A, B, C, loc);
                    return central_µ4 / (this.standardDeviation(A, B, C, loc) ** 4);
                },
                median: function (A, B, C, loc) {
                    return dists.burr_4p.measurements.ppf(0.5, A, B, C, loc);
                },
                mode: function (A, B, C, loc) {
                    return loc + A * ((B - 1) / (B * C + 1)) ** (1 / B);
                },
            },
        }
    }
}
console.log(dists.burr_4p.measurements.stats.mean(10, 6, 5, 100))
console.log(dists.burr_4p.measurements.stats.variance(10, 6, 5, 100))
console.log(dists.burr_4p.measurements.stats.standardDeviation(10, 6, 5, 100))
console.log(dists.burr_4p.measurements.stats.skewness(10, 6, 5, 100))
console.log(dists.burr_4p.measurements.stats.kurtosis(10, 6, 5, 100))
// console.log(dists.burr_4p.measurements.stats.median(10, 6, 5, 100))
console.log(dists.burr_4p.measurements.stats.mode(10, 6, 5, 100))

// mean_value: 107.24022098073651
// variance_value: 2.3166295234263785
// standard_deviation_value: 1.5220478058938813
// skewness_value: -0.09707354093711192
// kurtosis_value: 3.097969236698992
// median_value: 107.27865319910373
// mode_value: 107.37793319357775