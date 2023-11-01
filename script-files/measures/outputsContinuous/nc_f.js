jStat = require("../node_modules/jstat");

dists = {
    non_central_f: {
        measurements: {
            nonCentralMoments: function (k, lambda, n1, n2) {
                let result;
                switch (k) {
                    case 1: result = (n2 / n1) * ((n1 + lambda) / (n2 - 2)); break;
                    case 2: result = ((n2 / n1) ** 2 * (1 / ((n2 - 2) * (n2 - 4))) * (lambda ** 2 + (2 * lambda + n1) * (n1 + 2))); break;
                    case 3: result = ((n2 / n1) ** 3 * (1 / ((n2 - 2) * (n2 - 4) * (n2 - 6))) * (lambda ** 3 + 3 * (n1 + 4) * lambda ** 2 + (3 * lambda + n1) * (n1 + 4) * (n1 + 2))); break;
                    case 4: result = ((n2 / n1) ** 4 * (1 / ((n2 - 2) * (n2 - 4) * (n2 - 6) * (n2 - 8))) * (lambda ** 4 + 4 * (n1 + 6) * lambda ** 3 + 6 * (n1 + 6) * (n1 + 4) * lambda ** 2 + (4 * lambda + n1) * (n1 + 2) * (n1 + 4) * (n1 + 6))); break;
                };
                return result
            },
            centralMoments: function (k, lambda, n1, n2) {
                const miu1 = this.nonCentralMoments(1, lambda, n1, n2);
                const miu2 = this.nonCentralMoments(2, lambda, n1, n2);
                const miu3 = this.nonCentralMoments(3, lambda, n1, n2);
                const miu4 = this.nonCentralMoments(4, lambda, n1, n2);

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
                mean: function (lambda, n1, n2) {
                    const miu1 = dists.nc_f.measurements.nonCentralMoments(1, lambda, n1, n2);
                    return miu1;
                },
                variance: function (lambda, n1, n2) {
                    const miu1 = dists.nc_f.measurements.nonCentralMoments(1, lambda, n1, n2);
                    const miu2 = dists.nc_f.measurements.nonCentralMoments(2, lambda, n1, n2);
                    return miu2 - miu1 ** 2;
                },
                standardDeviation: function (lambda, n1, n2) {
                    return Math.sqrt(this.variance(lambda, n1, n2));
                },
                skewness: function (lambda, n1, n2) {
                    const central_miu3 = dists.nc_f.measurements.centralMoments(3, lambda, n1, n2);
                    return central_miu3 / (this.standardDeviation(lambda, n1, n2) ** 3);
                },
                kurtosis: function (lambda, n1, n2) {
                    const central_miu4 = dists.nc_f.measurements.centralMoments(4, lambda, n1, n2);
                    return central_miu4 / (this.standardDeviation(lambda, n1, n2) ** 4);
                },
                median: function (lambda, n1, n2) {
                    return dists.nc_f.measurements.ppf(0.5, lambda, n1, n2);
                },
                mode: function (lambda, n1, n2) {
                    return undefined;
                },
            },
        }
    }
}
console.log(dists.nc_f.measurements.stats.mean(100, 15, 57))
console.log(dists.nc_f.measurements.stats.variance(100, 15, 57))
console.log(dists.nc_f.measurements.stats.standardDeviation(100, 15, 57))
console.log(dists.nc_f.measurements.stats.skewness(100, 15, 57))
console.log(dists.nc_f.measurements.stats.kurtosis(100, 15, 57))
// console.log(dists.nc_f.measurements.stats.median(100, 15, 57))
console.log(dists.nc_f.measurements.stats.mode(100, 15, 57))

// mean_value: 7.945454545454544
// variance_value: 4.512359270232345
// standard_deviation_value: 2.124231454016333
// skewness_value: 0.8222489907196768
// kurtosis_value: 4.298545918660361
// median_value: None
// mode_value: None