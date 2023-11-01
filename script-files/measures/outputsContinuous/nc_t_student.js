jStat = require("../node_modules/jstat");

dists = {
    non_central_t_student: {
        measurements: {
            nonCentralMoments: function (k, lambda, n, loc, scale) {
                let result;
                switch (k) {
                    case 1: result = lambda * Math.sqrt(n / 2) * jStat.gammafn((n - 1) / 2) / jStat.gammafn(n / 2); break;
                    case 2: result = n * (1 + lambda * lambda) / (n - 2); break;
                    case 3: result = (n ** 1.5 * Math.sqrt(2) * jStat.gammafn((n - 3) / 2) * lambda * (3 + lambda * lambda)) / (4 * jStat.gammafn(n / 2)); break;
                    case 4: result = (n * n * (lambda ** 4 + 6 * lambda ** 2 + 3)) / ((n - 2) * (n - 4)); break;
                };
                return result
            },
            centralMoments: function (k, lambda, n, loc, scale) {
                const miu1 = this.nonCentralMoments(1, lambda, n, loc, scale);
                const miu2 = this.nonCentralMoments(2, lambda, n, loc, scale);
                const miu3 = this.nonCentralMoments(3, lambda, n, loc, scale);
                const miu4 = this.nonCentralMoments(4, lambda, n, loc, scale);

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
                mean: function (lambda, n, loc, scale) {
                    const miu1 = dists.NON_CENTRAL_T_STUDENT.measurements.nonCentralMoments(1, lambda, n, loc, scale);
                    return loc + scale * miu1;
                },
                variance: function (lambda, n, loc, scale) {
                    const miu1 = dists.NON_CENTRAL_T_STUDENT.measurements.nonCentralMoments(1, lambda, n, loc, scale);
                    const miu2 = dists.NON_CENTRAL_T_STUDENT.measurements.nonCentralMoments(2, lambda, n, loc, scale);
                    return (scale ** 2) * (miu2 - miu1 ** 2);
                },
                standardDeviation: function (lambda, n, loc, scale) {
                    return Math.sqrt(this.variance(lambda, n, loc, scale));
                },
                skewness: function (lambda, n, loc, scale) {
                    const miu1 = dists.NON_CENTRAL_T_STUDENT.measurements.nonCentralMoments(1, lambda, n, loc, scale);
                    const miu2 = dists.NON_CENTRAL_T_STUDENT.measurements.nonCentralMoments(2, lambda, n, loc, scale);
                    const central_miu3 = dists.NON_CENTRAL_T_STUDENT.measurements.centralMoments(3, lambda, n, loc, scale);
                    const std = Math.sqrt(miu2 - miu1 ** 2);
                    return central_miu3 / (std ** 3);
                },
                kurtosis: function (lambda, n, loc, scale) {
                    const miu1 = dists.NON_CENTRAL_T_STUDENT.measurements.nonCentralMoments(1, lambda, n, loc, scale);
                    const miu2 = dists.NON_CENTRAL_T_STUDENT.measurements.nonCentralMoments(2, lambda, n, loc, scale);
                    const central_miu4 = dists.NON_CENTRAL_T_STUDENT.measurements.centralMoments(4, lambda, n, loc, scale);
                    const std = Math.sqrt(miu2 - miu1 ** 2);
                    return central_miu4 / (std ** 4);
                },
                median: function (lambda, n, loc, scale) {
                    return dists.NON_CENTRAL_T_STUDENT.measurements.ppf(0.5, lambda, n, loc, scale);
                },
                mode: function (lambda, n, loc, scale) {
                    return undefined;
                },
            },
        }
    }
}
console.log(dists.NON_CENTRAL_T_STUDENT.measurements.stats.mean(2, 15, 10, 2))
console.log(dists.NON_CENTRAL_T_STUDENT.measurements.stats.variance(2, 15, 10, 2))
console.log(dists.NON_CENTRAL_T_STUDENT.measurements.stats.standardDeviation(2, 15, 10, 2))
console.log(dists.NON_CENTRAL_T_STUDENT.measurements.stats.skewness(2, 15, 10, 2))
console.log(dists.NON_CENTRAL_T_STUDENT.measurements.stats.kurtosis(2, 15, 10, 2))
// console.log(dists.NON_CENTRAL_T_STUDENT.measurements.stats.median(2, 15, 10, 2))
console.log(dists.NON_CENTRAL_T_STUDENT.measurements.stats.mode(2, 15, 10, 2))

// mean_value: 14.214929421210748
// variance_value: 5.311293051135106
// standard_deviation_value: 2.3046242754807356
// skewness_value: 0.4478135202004284
// kurtosis_value: 3.8402535036763026
// median_value: None
// mode_value: None