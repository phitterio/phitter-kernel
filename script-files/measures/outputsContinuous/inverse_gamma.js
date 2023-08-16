jStat = require("../node_modules/jstat");

dists = {
    inverse_gamma: {
        measurements: {
            nonCentralMoments: function (k, alpha, beta) {
                let result;
                switch (k) {
                    case 1: result = (beta**k)/((alpha-1)); break;
                    case 2: result = (beta**k)/((alpha-1)*(alpha-2)); break;
                    case 3: result = (beta**k)/((alpha-1)*(alpha-2)*(alpha-3)); break;
                    case 4: result = (beta**k)/((alpha-1)*(alpha-2)*(alpha-3)*(alpha-4)); break;
                };
                return result;
            },
            centralMoments: function (k, alpha, beta) {
                const µ1 = this.nonCentralMoments(1, alpha, beta);
                const µ2 = this.nonCentralMoments(2, alpha, beta);
                const µ3 = this.nonCentralMoments(3, alpha, beta);
                const µ4 = this.nonCentralMoments(4, alpha, beta);

                let result;
                switch (k) {
                    case 1: result = 0; break;
                    case 2: result = µ2 - µ1 ** 2; break;
                    case 3: result = µ3 - 3 * µ1 * µ2 + 2 * µ1 ** 3; break;
                    case 4: result = µ4 - 4 * µ1 * µ3 + 6 * (µ1 ** 2) * µ2 - 3 * (µ1 ** 4); break;
                };
                return result;
            },
            stats: {
                mean: function (alpha, beta) {
                    const µ1 = dists.inverse_gamma.measurements.nonCentralMoments(1, alpha, beta);
                    return µ1;
                },
                variance: function (alpha, beta) {
                    const µ1 = dists.inverse_gamma.measurements.nonCentralMoments(1, alpha, beta);
                    const µ2 = dists.inverse_gamma.measurements.nonCentralMoments(2, alpha, beta);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (alpha, beta) {
                    return Math.sqrt(this.variance(alpha, beta));
                },
                skewness: function (alpha, beta) {
                    const central_µ3 = dists.inverse_gamma.measurements.centralMoments(3, alpha, beta);
                    return central_µ3 / (this.standardDeviation(alpha, beta) ** 3);
                },
                kurtosis: function (alpha, beta) {
                    const central_µ4 = dists.inverse_gamma.measurements.centralMoments(4, alpha, beta);
                    return central_µ4 / (this.standardDeviation (alpha, beta) ** 4);
                },
                median: function (alpha, beta) {
                    return dists.inverse_gamma.measurements.ppf(0.5, alpha, beta);
                },
                mode: function (alpha, beta) {
                    return beta/(alpha+1);
                },
            },
        }
    }
}
console.log(dists.inverse_gamma.measurements.stats.mean(5, 1))
console.log(dists.inverse_gamma.measurements.stats.variance(5, 1))
console.log(dists.inverse_gamma.measurements.stats.standardDeviation(5, 1))
console.log(dists.inverse_gamma.measurements.stats.skewness(5, 1))
console.log(dists.inverse_gamma.measurements.stats.kurtosis(5, 1))
// console.log(dists.inverse_gamma.measurements.stats.median(5, 1))
console.log(dists.inverse_gamma.measurements.stats.mode(5, 1))

// mean_value: 0.25
// variance_value: 0.02083333333333333
// standard_deviation_value: 0.14433756729740643
// skewness_value: 3.4641016151377557
// kurtosis_value: 45.00000000000002
// median_value: 0.21409109556455425
// mode_value: 0.16666666666666666