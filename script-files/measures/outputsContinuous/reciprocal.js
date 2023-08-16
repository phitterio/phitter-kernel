jStat = require("../node_modules/jstat");

dists = {
    reciprocal: {
        measurements: {
            nonCentralMoments: function (k, a, b) {
                return (b ** k - a ** k) / (k * (Math.log(b) - Math.log(a)))
            },
            centralMoments: function (k, a, b) {
                const µ1 = this.nonCentralMoments(1, a, b);
                const µ2 = this.nonCentralMoments(2, a, b);
                const µ3 = this.nonCentralMoments(3, a, b);
                const µ4 = this.nonCentralMoments(4, a, b);

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
                mean: function (a, b) {
                    const µ1 = dists.reciprocal.measurements.nonCentralMoments(1, a, b);
                    return µ1;
                },
                variance: function (a, b) {
                    const µ1 = dists.reciprocal.measurements.nonCentralMoments(1, a, b);
                    const µ2 = dists.reciprocal.measurements.nonCentralMoments(2, a, b);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (a, b) {
                    return Math.sqrt(this.variance(a, b));
                },
                skewness: function (a, b) {
                    const central_µ3 = dists.reciprocal.measurements.centralMoments(3, a, b);
                    return central_µ3 / (this.standardDeviation(a, b) ** 3);
                },
                kurtosis: function (a, b) {
                    const central_µ4 = dists.reciprocal.measurements.centralMoments(4, a, b);
                    return central_µ4 / (this.standardDeviation(a, b) ** 4);
                },
                median: function (a, b) {
                    return dists.reciprocal.measurements.ppf(0.5, a, b);
                },
                mode: function (a, b) {
                    return a;
                },
            },
        }
    }
}
console.log(dists.reciprocal.measurements.stats.mean(20, 100))
console.log(dists.reciprocal.measurements.stats.variance(20, 100))
console.log(dists.reciprocal.measurements.stats.standardDeviation(20, 100))
console.log(dists.reciprocal.measurements.stats.skewness(20, 100))
console.log(dists.reciprocal.measurements.stats.kurtosis(20, 100))
// console.log(dists.reciprocal.measurements.stats.median(20, 100))
console.log(dists.reciprocal.measurements.stats.mode(20, 100))

// mean_value: 49.70679476476893
// variance_value: 511.642240099276
// standard_deviation_value: 22.61951016488368
// skewness_value: 0.5482226050475848
// kurtosis_value: 2.1295341717725487
// median_value: 44.72135954999579
// mode_value: 20