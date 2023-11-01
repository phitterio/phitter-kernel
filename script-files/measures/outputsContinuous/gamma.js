jStat = require("../node_modules/jstat");

dists = {
    gamma: {
        measurements: {
            nonCentralMoments: function (k, alpha, beta) {
                return beta ** k * (jStat.gammafn(k + alpha) / (jStat.gammafn(alpha)))
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
                    const miu1 = dists.gamma.measurements.nonCentralMoments(1, alpha, beta);
                    return miu1;
                },
                variance: function (alpha, beta) {
                    const miu1 = dists.gamma.measurements.nonCentralMoments(1, alpha, beta);
                    const miu2 = dists.gamma.measurements.nonCentralMoments(2, alpha, beta);
                    return miu2 - miu1 ** 2;
                },
                standardDeviation: function (alpha, beta) {
                    return Math.sqrt(this.variance(alpha, beta));
                },
                skewness: function (alpha, beta) {
                    const central_miu3 = dists.gamma.measurements.centralMoments(3, alpha, beta);
                    return central_miu3 / (this.standardDeviation(alpha, beta) ** 3);
                },
                kurtosis: function (alpha, beta) {
                    const central_miu4 = dists.gamma.measurements.centralMoments(4, alpha, beta);
                    return central_miu4 / (this.standardDeviation(alpha, beta) ** 4);
                },
                median: function (alpha, beta) {
                    return dists.gamma.measurements.ppf(0.5, alpha, beta);
                },
                mode: function (alpha, beta) {
                    return beta * (alpha - 1);
                },
            },
        }
    }
}
console.log(dists.gamma.measurements.stats.mean(3, 10))
console.log(dists.gamma.measurements.stats.variance(3, 10))
console.log(dists.gamma.measurements.stats.standardDeviation(3, 10))
console.log(dists.gamma.measurements.stats.skewness(3, 10))
console.log(dists.gamma.measurements.stats.kurtosis(3, 10))
// console.log(dists.gamma.measurements.stats.median(3, 10))
console.log(dists.gamma.measurements.stats.mode(3, 10))

// mean_value: 30
// variance_value: 300
// standard_deviation_value: 17.320508075688775
// skewness_value: 1.1547005383792512
// kurtosis_value: 4.999999999999998
// median_value: 26.74060313723561
// mode_value: 20