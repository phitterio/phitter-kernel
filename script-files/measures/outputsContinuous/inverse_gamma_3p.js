jStat = require("../node_modules/jstat");

dists = {
    inverse_gamma_3p: {
        measurements: {
            nonCentralMoments: function (k, alpha, beta, loc) {
                let result;
                switch (k) {
                    case 1: result = (beta ** k) / ((alpha - 1)); break;
                    case 2: result = (beta ** k) / ((alpha - 1) * (alpha - 2)); break;
                    case 3: result = (beta ** k) / ((alpha - 1) * (alpha - 2) * (alpha - 3)); break;
                    case 4: result = (beta ** k) / ((alpha - 1) * (alpha - 2) * (alpha - 3) * (alpha - 4)); break;
                };
                return result
            },
            centralMoments: function (k, alpha, beta, loc) {
                const miu1 = this.nonCentralMoments(1, alpha, beta, loc);
                const miu2 = this.nonCentralMoments(2, alpha, beta, loc);
                const miu3 = this.nonCentralMoments(3, alpha, beta, loc);
                const miu4 = this.nonCentralMoments(4, alpha, beta, loc);

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
                mean: function (alpha, beta, loc) {
                    const miu1 = dists.inverse_gamma_3p.measurements.nonCentralMoments(1, alpha, beta, loc);
                    return loc + miu1;
                },
                variance: function (alpha, beta, loc) {
                    const miu1 = dists.inverse_gamma_3p.measurements.nonCentralMoments(1, alpha, beta, loc);
                    const miu2 = dists.inverse_gamma_3p.measurements.nonCentralMoments(2, alpha, beta, loc);
                    return miu2 - miu1 ** 2;
                },
                standardDeviation: function (alpha, beta, loc) {
                    return Math.sqrt(this.variance(alpha, beta, loc));
                },
                skewness: function (alpha, beta, loc) {
                    const central_miu3 = dists.inverse_gamma_3p.measurements.centralMoments(3, alpha, beta, loc);
                    return central_miu3 / (this.standardDeviation(alpha, beta, loc) ** 3);
                },
                kurtosis: function (alpha, beta, loc) {
                    const central_miu4 = dists.inverse_gamma_3p.measurements.centralMoments(4, alpha, beta, loc);
                    return central_miu4 / (this.standardDeviation(alpha, beta, loc) ** 4);
                },
                median: function (alpha, beta, loc) {
                    return dists.inverse_gamma_3p.measurements.ppf(0.5, alpha, beta, loc);
                },
                mode: function (alpha, beta, loc) {
                    return beta / (alpha + 1) + loc;
                },
            },
        }
    }
}
console.log(dists.inverse_gamma_3p.measurements.stats.mean(5, 10, 100))
console.log(dists.inverse_gamma_3p.measurements.stats.variance(5, 10, 100))
console.log(dists.inverse_gamma_3p.measurements.stats.standardDeviation(5, 10, 100))
console.log(dists.inverse_gamma_3p.measurements.stats.skewness(5, 10, 100))
console.log(dists.inverse_gamma_3p.measurements.stats.kurtosis(5, 10, 100))
// console.log(dists.inverse_gamma_3p.measurements.stats.median(5, 10, 100))
console.log(dists.inverse_gamma_3p.measurements.stats.mode(5, 10, 100))

// mean_value: 102.5
// variance_value: 2.083333333333334
// standard_deviation_value: 1.4433756729740645
// skewness_value: 3.4641016151377495
// kurtosis_value: 44.999999999999986
// median_value: 102.14091095564554
// mode_value: 101.66666666666667