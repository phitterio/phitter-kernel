jStat = require("../node_modules/jstat");

dists = {
    nc_t_student: {
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
                const µ1 = this.nonCentralMoments(1, lambda, n, loc, scale);
                const µ2 = this.nonCentralMoments(2, lambda, n, loc, scale);
                const µ3 = this.nonCentralMoments(3, lambda, n, loc, scale);
                const µ4 = this.nonCentralMoments(4, lambda, n, loc, scale);

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
                mean: function (lambda, n, loc, scale) {
                    const µ1 = dists.nc_t_student.measurements.nonCentralMoments(1, lambda, n, loc, scale);
                    return loc + scale * µ1;
                },
                variance: function (lambda, n, loc, scale) {
                    const µ1 = dists.nc_t_student.measurements.nonCentralMoments(1, lambda, n, loc, scale);
                    const µ2 = dists.nc_t_student.measurements.nonCentralMoments(2, lambda, n, loc, scale);
                    return (scale ** 2) * (µ2 - µ1 ** 2);
                },
                standardDeviation: function (lambda, n, loc, scale) {
                    return Math.sqrt(this.variance(lambda, n, loc, scale));
                },
                skewness: function (lambda, n, loc, scale) {
                    const µ1 = dists.nc_t_student.measurements.nonCentralMoments(1, lambda, n, loc, scale);
                    const µ2 = dists.nc_t_student.measurements.nonCentralMoments(2, lambda, n, loc, scale);
                    const central_µ3 = dists.nc_t_student.measurements.centralMoments(3, lambda, n, loc, scale);
                    const std = Math.sqrt(µ2 - µ1 ** 2);
                    return central_µ3 / (std ** 3);
                },
                kurtosis: function (lambda, n, loc, scale) {
                    const µ1 = dists.nc_t_student.measurements.nonCentralMoments(1, lambda, n, loc, scale);
                    const µ2 = dists.nc_t_student.measurements.nonCentralMoments(2, lambda, n, loc, scale);
                    const central_µ4 = dists.nc_t_student.measurements.centralMoments(4, lambda, n, loc, scale);
                    const std = Math.sqrt(µ2 - µ1 ** 2);
                    return central_µ4 / (std ** 4);
                },
                median: function (lambda, n, loc, scale) {
                    return dists.nc_t_student.measurements.ppf(0.5, lambda, n, loc, scale);
                },
                mode: function (lambda, n, loc, scale) {
                    return undefined;
                },
            },
        }
    }
}
console.log(dists.nc_t_student.measurements.stats.mean(2, 15, 10, 2))
console.log(dists.nc_t_student.measurements.stats.variance(2, 15, 10, 2))
console.log(dists.nc_t_student.measurements.stats.standardDeviation(2, 15, 10, 2))
console.log(dists.nc_t_student.measurements.stats.skewness(2, 15, 10, 2))
console.log(dists.nc_t_student.measurements.stats.kurtosis(2, 15, 10, 2))
// console.log(dists.nc_t_student.measurements.stats.median(2, 15, 10, 2))
console.log(dists.nc_t_student.measurements.stats.mode(2, 15, 10, 2))

// mean_value: 14.214929421210748
// variance_value: 5.311293051135106
// standard_deviation_value: 2.3046242754807356
// skewness_value: 0.4478135202004284
// kurtosis_value: 3.8402535036763026
// median_value: None
// mode_value: None