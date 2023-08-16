jStat = require("../node_modules/jstat");

dists = {
    gamma: {
        measurements: {
            nonCentralMoments: function (k, alpha, beta) {
                return beta ** k * (jStat.gammafn(k + alpha) / (jStat.gammafn(alpha)))
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
                    const µ1 = dists.gamma.measurements.nonCentralMoments(1, alpha, beta);
                    return µ1;
                },
                variance: function (alpha, beta) {
                    const µ1 = dists.gamma.measurements.nonCentralMoments(1, alpha, beta);
                    const µ2 = dists.gamma.measurements.nonCentralMoments(2, alpha, beta);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (alpha, beta) {
                    return Math.sqrt(this.variance(alpha, beta));
                },
                skewness: function (alpha, beta) {
                    const central_µ3 = dists.gamma.measurements.centralMoments(3, alpha, beta);
                    return central_µ3 / (this.standardDeviation(alpha, beta) ** 3);
                },
                kurtosis: function (alpha, beta) {
                    const central_µ4 = dists.gamma.measurements.centralMoments(4, alpha, beta);
                    return central_µ4 / (this.standardDeviation(alpha, beta) ** 4);
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