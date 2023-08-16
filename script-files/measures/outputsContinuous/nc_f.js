jStat = require("../node_modules/jstat");

dists = {
    nc_f: {
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
                const µ1 = this.nonCentralMoments(1, lambda, n1, n2);
                const µ2 = this.nonCentralMoments(2, lambda, n1, n2);
                const µ3 = this.nonCentralMoments(3, lambda, n1, n2);
                const µ4 = this.nonCentralMoments(4, lambda, n1, n2);

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
                mean: function (lambda, n1, n2) {
                    const µ1 = dists.nc_f.measurements.nonCentralMoments(1, lambda, n1, n2);
                    return µ1;
                },
                variance: function (lambda, n1, n2) {
                    const µ1 = dists.nc_f.measurements.nonCentralMoments(1, lambda, n1, n2);
                    const µ2 = dists.nc_f.measurements.nonCentralMoments(2, lambda, n1, n2);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (lambda, n1, n2) {
                    return Math.sqrt(this.variance(lambda, n1, n2));
                },
                skewness: function (lambda, n1, n2) {
                    const central_µ3 = dists.nc_f.measurements.centralMoments(3, lambda, n1, n2);
                    return central_µ3 / (this.standardDeviation(lambda, n1, n2) ** 3);
                },
                kurtosis: function (lambda, n1, n2) {
                    const central_µ4 = dists.nc_f.measurements.centralMoments(4, lambda, n1, n2);
                    return central_µ4 / (this.standardDeviation(lambda, n1, n2) ** 4);
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