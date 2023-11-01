jStat = require("../node_modules/jstat");

dists = {
    weibull: {
        measurements: {
            nonCentralMoments: function (k, alpha, beta) {
                return (beta ** k) * jStat.gammafn(1 + k / alpha)
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
                return result
            },
            stats: {
                mean: function (alpha, beta) {
                    const miu1 = dists.weibull.measurements.nonCentralMoments(1, alpha, beta);
                    return miu1;
                },
                variance: function (alpha, beta) {
                    const miu1 = dists.weibull.measurements.nonCentralMoments(1, alpha, beta);
                    const miu2 = dists.weibull.measurements.nonCentralMoments(2, alpha, beta);
                    return miu2 - miu1 ** 2;
                },
                standardDeviation: function (alpha, beta) {
                    return Math.sqrt(this.variance(alpha, beta));
                },
                skewness: function (alpha, beta) {
                    const central_miu3 = dists.weibull.measurements.centralMoments(3, alpha, beta);
                    return central_miu3 / (this.standardDeviation(alpha, beta) ** 3);
                },
                kurtosis: function (alpha, beta) {
                    const central_miu4 = dists.weibull.measurements.centralMoments(4, alpha, beta);
                    return central_miu4 / (this.standardDeviation(alpha, beta) ** 4);
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