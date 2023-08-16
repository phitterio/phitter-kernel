jStat = require("../node_modules/jstat");

dists = {
    loglogistic_3p: {
        measurements: {
            nonCentralMoments: function (k, alpha, beta, loc) {
                return alpha ** k * (k * Math.PI / beta) / (Math.sin(k * Math.PI / beta))
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
                    const µ1 = dists.loglogistic_3p.measurements.nonCentralMoments(1, alpha, beta, loc);
                    return loc + µ1;
                },
                variance: function (alpha, beta, loc) {
                    const µ1 = dists.loglogistic_3p.measurements.nonCentralMoments(1, alpha, beta, loc);
                    const µ2 = dists.loglogistic_3p.measurements.nonCentralMoments(2, alpha, beta, loc);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (alpha, beta, loc) {
                    return Math.sqrt(this.variance(alpha, beta, loc));
                },
                skewness: function (alpha, beta, loc) {
                    const central_µ3 = dists.loglogistic_3p.measurements.centralMoments(3, alpha, beta, loc);
                    return central_µ3 / (this.standardDeviation(alpha, beta, loc) ** 3);
                },
                kurtosis: function (alpha, beta, loc) {
                    const central_µ4 = dists.loglogistic_3p.measurements.centralMoments(4, alpha, beta, loc);
                    return central_µ4 / (this.standardDeviation(alpha, beta, loc) ** 4);
                },
                median: function (alpha, beta, loc) {
                    return dists.loglogistic_3p.measurements.ppf(0.5, alpha, beta, loc);
                },
                mode: function (alpha, beta, loc) {
                    return loc + alpha * ((beta - 1) / (beta + 1)) ** (1 / beta);
                },
            },
        }
    }
}
console.log(dists.loglogistic_3p.measurements.stats.mean(5, 3, 100))
console.log(dists.loglogistic_3p.measurements.stats.variance(5, 3, 100))
console.log(dists.loglogistic_3p.measurements.stats.standardDeviation(5, 3, 100))
console.log(dists.loglogistic_3p.measurements.stats.skewness(5, 3, 100))
console.log(dists.loglogistic_3p.measurements.stats.kurtosis(5, 3, 100))
// console.log(dists.loglogistic_3p.measurements.stats.median(5, 3, 100))
console.log(dists.loglogistic_3p.measurements.stats.mode(5, 3, 100))

// mean_value: 106.04599788078073
// variance_value: 23.905888433402232
// standard_deviation_value: 4.889364829239298
// skewness_value: 2.742292882738054e+16
// kurtosis_value: -1.3564049758253683e+17
// median_value: 105
// mode_value: 103.9685026299205