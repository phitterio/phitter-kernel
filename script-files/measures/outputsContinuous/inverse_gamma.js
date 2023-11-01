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
                const miu1 = this.nonCentralMoments(1, alpha, beta);
                const miu2 = this.nonCentralMoments(2, alpha, beta);
                const miu3 = this.nonCentralMoments(3, alpha, beta);
                const miu4 = this.nonCentralMoments(4, alpha, beta);

                let result;
                switch (k) {
                    case 1: result = 0; break;
                    case 2: result = miu2 - miu1 ** 2; break;
                    case 3: result = miu3 - 3 * miu1 * miu2 + 2 * miu1 ** 3; break;
                    case 4: result = miu4 - 4 * miu1 * miu3 + 6 * (miu1 ** 2) * miu2 - 3 * (miu1 ** 4); break;
                };
                return result;
            },
            stats: {
                mean: function (alpha, beta) {
                    const miu1 = dists.inverse_gamma.measurements.nonCentralMoments(1, alpha, beta);
                    return miu1;
                },
                variance: function (alpha, beta) {
                    const miu1 = dists.inverse_gamma.measurements.nonCentralMoments(1, alpha, beta);
                    const miu2 = dists.inverse_gamma.measurements.nonCentralMoments(2, alpha, beta);
                    return miu2 - miu1 ** 2;
                },
                standardDeviation: function (alpha, beta) {
                    return Math.sqrt(this.variance(alpha, beta));
                },
                skewness: function (alpha, beta) {
                    const central_miu3 = dists.inverse_gamma.measurements.centralMoments(3, alpha, beta);
                    return central_miu3 / (this.standardDeviation(alpha, beta) ** 3);
                },
                kurtosis: function (alpha, beta) {
                    const central_miu4 = dists.inverse_gamma.measurements.centralMoments(4, alpha, beta);
                    return central_miu4 / (this.standardDeviation (alpha, beta) ** 4);
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