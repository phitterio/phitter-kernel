jStat = require("../node_modules/jstat");

dists = {
    loglogistic: {
        measurements: {
            nonCentralMoments: function (k, alpha, beta) {
                return alpha ** k * (k * Math.PI / beta) / (Math.sin(k * Math.PI / beta))
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
                    const µ1 = dists.loglogistic.measurements.nonCentralMoments(1, alpha, beta);
                    return µ1;
                },
                variance: function (alpha, beta) {
                    const µ1 = dists.loglogistic.measurements.nonCentralMoments(1, alpha, beta);
                    const µ2 = dists.loglogistic.measurements.nonCentralMoments(2, alpha, beta);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (alpha, beta) {
                    return Math.sqrt(this.variance(alpha, beta));
                },
                skewness: function (alpha, beta) {
                    const central_µ3 = dists.loglogistic.measurements.centralMoments(3, alpha, beta);
                    return central_µ3 / (this.standardDeviation(alpha, beta) ** 3);
                },
                kurtosis: function (alpha, beta) {
                    const central_µ4 = dists.loglogistic.measurements.centralMoments(4, alpha, beta);
                    return central_µ4 / (this.standardDeviation(alpha, beta) ** 4);
                },
                median: function (alpha, beta) {
                    return dists.loglogistic.measurements.ppf(0.5, alpha, beta);
                },
                mode: function (alpha, beta) {
                    return alpha * ((beta - 1) / (beta + 1)) ** (1 / beta);
                },
            },
        }
    }
}
console.log(dists.loglogistic.measurements.stats.mean(100, 3))
console.log(dists.loglogistic.measurements.stats.variance(100, 3))
console.log(dists.loglogistic.measurements.stats.standardDeviation(100, 3))
console.log(dists.loglogistic.measurements.stats.skewness(100, 3))
console.log(dists.loglogistic.measurements.stats.kurtosis(100, 3))
// console.log(dists.loglogistic.measurements.stats.median(100, 3))
console.log(dists.loglogistic.measurements.stats.mode(100, 3))

// mean_value: 120.91995761561452
// variance_value: 9562.35537336089
// standard_deviation_value: 97.78729658478595
// skewness_value: 2.7422928827380556e+16
// kurtosis_value: -1.3564049758253694e+17
// median_value: 100
// mode_value: 79.37005259840998