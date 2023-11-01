jStat = require("../node_modules/jstat");

dists = {
    dagum_4p: {
        measurements: {
            nonCentralMoments: function (k, a, b, p, loc) {
                return (b ** k) * (p) * (jStat.gammafn((a * p + k) / a) * jStat.gammafn((a - k) / a) / jStat.gammafn(((a * p + k) / a) + ((a - k) / a)))
            },
            centralMoments: function (k, a, b, p, loc) {
                const miu1 = this.nonCentralMoments(1, a, b, p, loc);
                const miu2 = this.nonCentralMoments(2, a, b, p, loc);
                const miu3 = this.nonCentralMoments(3, a, b, p, loc);
                const miu4 = this.nonCentralMoments(4, a, b, p, loc);

                let result;
                switch (k) {
                    case 1: result = 0; break;
                    case 2: result = miu2 - miu1 ** 2; break;
                    case 3: result = miu3 - 3 * miu1 * miu2 + 2 * miu1 ** 3; break;
                    case 4: result = miu4 - 4 * miu1 * miu3 + 6 * (miu1 ** 2) * miu2 - 3 * (miu1 ** 4); break;
                };
                return result
            },
            stats: {
                mean: function (a, b, p, loc) {
                    const miu1 = dists.dagum_4p.measurements.nonCentralMoments(1, a, b, p, loc);
                    return loc + miu1;
                },
                variance: function (a, b, p, loc) {
                    const miu1 = dists.dagum_4p.measurements.nonCentralMoments(1, a, b, p, loc);
                    const miu2 = dists.dagum_4p.measurements.nonCentralMoments(2, a, b, p, loc);
                    return miu2 - miu1 ** 2;
                },
                standardDeviation: function (a, b, p, loc) {
                    return Math.sqrt(this.variance(a, b, p, loc));
                },
                skewness: function (a, b, p, loc) {
                    const central_miu3 = dists.dagum_4p.measurements.centralMoments(3, a, b, p, loc);
                    return central_miu3 / (this.standardDeviation(a, b, p, loc) ** 3);
                },
                kurtosis: function (a, b, p, loc) {
                    const central_miu4 = dists.dagum_4p.measurements.centralMoments(4, a, b, p, loc);
                    return central_miu4 / (this.standardDeviation(a, b, p, loc) ** 4);
                },
                median: function (a, b, p, loc) {
                    return dists.dagum_4p.measurements.ppf(0.5, a, b, p, loc);
                },
                mode: function (a, b, p, loc) {
                    return loc + b * ((a * p - 1) / (a + 1)) ** (1 / a);
                },
            },
        }
    }
}
console.log(dists.dagum_4p.measurements.stats.mean(7, 2, 3, 100))
console.log(dists.dagum_4p.measurements.stats.variance(7, 2, 3, 100))
console.log(dists.dagum_4p.measurements.stats.standardDeviation(7, 2, 3, 100))
console.log(dists.dagum_4p.measurements.stats.skewness(7, 2, 3, 100))
console.log(dists.dagum_4p.measurements.stats.kurtosis(7, 2, 3, 100))
// console.log(dists.dagum_4p.measurements.stats.median(7, 2, 3, 100))
console.log(dists.dagum_4p.measurements.stats.mode(7, 2, 3, 100))

// mean_value: 102.53316584963297
// variance_value: 0.330915521516463
// standard_deviation_value: 0.5752525719338097
// skewness_value: 2.07640773242886
// kurtosis_value: 14.629714684287286
// median_value: 102.42451052557608
// mode_value: 102.27970456209519