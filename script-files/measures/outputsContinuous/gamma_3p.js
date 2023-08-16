jStat = require("../node_modules/jstat");

dists = {
    gamma_3p: {
        measurements: {
            nonCentralMoments: function (k, alpha, beta, loc) {
                return beta ** k * (jStat.gammafn(k + alpha) / (jStat.factorial(alpha - 1)))
            },
            centralMoments: function (k, alpha, beta, loc) {
                const µ1 = this.nonCentralMoments(1, alpha, beta, loc);
                const µ2 = this.nonCentralMoments(2, alpha, beta, loc);
                const µ3 = this.nonCentralMoments(3, alpha, beta, loc);
                const µ4 = this.nonCentralMoments(4, alpha, beta, loc);

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
                mean: function (alpha, beta, loc) {
                    const µ1 = dists.gamma_3p.measurements.nonCentralMoments(1, alpha, beta, loc);
                    return loc + µ1;
                },
                variance: function (alpha, beta, loc) {
                    const µ1 = dists.gamma_3p.measurements.nonCentralMoments(1, alpha, beta, loc);
                    const µ2 = dists.gamma_3p.measurements.nonCentralMoments(2, alpha, beta, loc);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (alpha, beta, loc) {
                    return Math.sqrt(this.variance(alpha, beta, loc));
                },
                skewness: function (alpha, beta, loc) {
                    const central_µ3 = dists.gamma_3p.measurements.centralMoments(3, alpha, beta, loc);
                    return central_µ3 / (this.standardDeviation(alpha, beta, loc) ** 3);
                },
                kurtosis: function (alpha, beta, loc) {
                    const central_µ4 = dists.gamma_3p.measurements.centralMoments(4, alpha, beta, loc);
                    return central_µ4 / (this.standardDeviation(alpha, beta, loc) ** 4);
                },
                median: function (alpha, beta, loc) {
                    return dists.gamma_3p.measurements.ppf(0.5, alpha, beta, loc);
                },
                mode: function (alpha, beta, loc) {
                    return beta * (alpha - 1) + loc;
                },
            },
        }
    }
}
console.log(dists.gamma_3p.measurements.stats.mean(25, 2, 100))
console.log(dists.gamma_3p.measurements.stats.variance(25, 2, 100))
console.log(dists.gamma_3p.measurements.stats.standardDeviation(25, 2, 100))
console.log(dists.gamma_3p.measurements.stats.skewness(25, 2, 100))
console.log(dists.gamma_3p.measurements.stats.kurtosis(25, 2, 100))
// console.log(dists.gamma_3p.measurements.stats.median(25, 2, 100))
console.log(dists.gamma_3p.measurements.stats.mode(25, 2, 100))

// mean_value: 150
// variance_value: 100.00000000000136
// standard_deviation_value: 10.000000000000068
// skewness_value: 0.3999999999998755
// kurtosis_value: 3.2400000000010305
// median_value: 149.33493673397683
// mode_value: 148