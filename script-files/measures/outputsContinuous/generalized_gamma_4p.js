jStat = require("../node_modules/jstat");

dists = {
    generalized_gamma_4p: {
        measurements: {
            nonCentralMoments: function (k, a, d, p, loc) {
                return (a ** k) * jStat.gammafn((d + k) / p) / jStat.gammafn((d) / p)
            },
            centralMoments: function (k, a, d, p, loc) {
                const miu1 = this.nonCentralMoments(1, a, d, p, loc);
                const miu2 = this.nonCentralMoments(2, a, d, p, loc);
                const miu3 = this.nonCentralMoments(3, a, d, p, loc);
                const miu4 = this.nonCentralMoments(4, a, d, p, loc);

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
                mean: function (a, d, p, loc) {
                    const miu1 = dists.generalized_gamma_4p.measurements.nonCentralMoments(1, a, d, p, loc);
                    return loc + miu1;
                },
                variance: function (a, d, p, loc) {
                    const miu1 = dists.generalized_gamma_4p.measurements.nonCentralMoments(1, a, d, p, loc);
                    const miu2 = dists.generalized_gamma_4p.measurements.nonCentralMoments(2, a, d, p, loc);
                    return miu2 - miu1 ** 2;
                },
                standardDeviation: function (a, d, p, loc) {
                    return Math.sqrt(this.variance(a, d, p, loc));
                },
                skewness: function (a, d, p, loc) {
                    const central_miu3 = dists.generalized_gamma_4p.measurements.centralMoments(3, a, d, p, loc);
                    return central_miu3 / (this.standardDeviation(a, d, p, loc) ** 3);
                },
                kurtosis: function (a, d, p, loc) {
                    const central_miu4 = dists.generalized_gamma_4p.measurements.centralMoments(4, a, d, p, loc);
                    return central_miu4 / (this.standardDeviation(a, d, p, loc) ** 4);
                },
                median: function (a, d, p, loc) {
                    return dists.generalized_gamma_4p.measurements.ppf(0.5, a, d, p, loc);
                },
                mode: function (a, d, p, loc) {
                    return loc + a * ((d - 1) / p) ** (1 / p);
                },
            },
        }
    }
}
console.log(dists.generalized_gamma_4p.measurements.stats.mean(2, 6, 2, 30))
console.log(dists.generalized_gamma_4p.measurements.stats.variance(2, 6, 2, 30))
console.log(dists.generalized_gamma_4p.measurements.stats.standardDeviation(2, 6, 2, 30))
console.log(dists.generalized_gamma_4p.measurements.stats.skewness(2, 6, 2, 30))
console.log(dists.generalized_gamma_4p.measurements.stats.kurtosis(2, 6, 2, 30))
// console.log(dists.generalized_gamma_4p.measurements.stats.median(2, 6, 2, 30))
console.log(dists.generalized_gamma_4p.measurements.stats.mode(2, 6, 2, 30))

// mean_value: 33.32335097044784
// variance_value: 0.9553383272233837
// standard_deviation_value: 0.9774141022224837
// skewness_value: 0.39710489225219237
// kurtosis_value: 3.025111348684857
// median_value: 33.27051085533961
// mode_value: 33.16227766016838