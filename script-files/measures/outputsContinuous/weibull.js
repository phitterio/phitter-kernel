jStat = require("../node_modules/jstat");

dists = {
    weibull: {
        measurements: {
            nonCentralMoments: function (k, alpha, beta) {
                return (beta ** k) * jStat.gammafn(1 + k / alpha)
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
                return result
            },
            stats: {
                mean: function (alpha, beta) {
                    const µ1 = dists.weibull.measurements.nonCentralMoments(1, alpha, beta);
                    return µ1;
                },
                variance: function (alpha, beta) {
                    const µ1 = dists.weibull.measurements.nonCentralMoments(1, alpha, beta);
                    const µ2 = dists.weibull.measurements.nonCentralMoments(2, alpha, beta);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (alpha, beta) {
                    return Math.sqrt(this.variance(alpha, beta));
                },
                skewness: function (alpha, beta) {
                    const central_µ3 = dists.weibull.measurements.centralMoments(3, alpha, beta);
                    return central_µ3 / (this.standardDeviation(alpha, beta) ** 3);
                },
                kurtosis: function (alpha, beta) {
                    const central_µ4 = dists.weibull.measurements.centralMoments(4, alpha, beta);
                    return central_µ4 / (this.standardDeviation(alpha, beta) ** 4);
                },
                median: function (alpha, beta) {
                    return dists.weibull.measurements.ppf(0.5, alpha, beta);
                },
                mode: function (alpha, beta) {
                    return beta * ((alpha - 1) / alpha) ** (1 / alpha);
                },
            },
        }
    }
}
console.log(dists.weibull.measurements.stats.mean(7, 10))
console.log(dists.weibull.measurements.stats.variance(7, 10))
console.log(dists.weibull.measurements.stats.standardDeviation(7, 10))
console.log(dists.weibull.measurements.stats.skewness(7, 10))
console.log(dists.weibull.measurements.stats.kurtosis(7, 10))
// console.log(dists.weibull.measurements.stats.median(7, 10))
console.log(dists.weibull.measurements.stats.mode(7, 10))

// mean_value: 9.354375628925466
// variance_value: 2.470374243249225
// standard_deviation_value: 1.5717424226791186
// skewness_value: -0.4631896323164616
// kurtosis_value: 3.187182954977602
// median_value: 9.489881297143224
// mode_value: 9.782191779821828