jStat = require("../node_modules/jstat");

dists = {
    power_function: {
        measurements: {
            nonCentralMoments: function (k, alpha, a, b) {
                let result;
                switch (k) {
                    case 1: result = (a + b * alpha) / (alpha + 1); break;
                    case 2: result = (2 * a ** 2 + 2 * alpha * a * b + alpha * (alpha + 1) * b ** 2) / ((alpha + 1) * (alpha + 2)); break;
                    case 3: result = (6 * a ** 3 + 6 * a ** 2 * b * alpha + 3 * a * b ** 2 * alpha * (1 + alpha) + b ** 3 * alpha * (1 + alpha) * (2 + alpha)) / ((1 + alpha) * (2 + alpha) * (3 + alpha)); break;
                    case 4: result = (24 * a ** 4 + 24 * alpha * a ** 3 * b + 12 * alpha * (alpha + 1) * a ** 2 * b ** 2 + 4 * alpha * (alpha + 1) * (alpha + 2) * a * b ** 3 + alpha * (alpha + 1) * (alpha + 2) * (alpha + 3) * b ** 4) / ((alpha + 1) * (alpha + 2) * (alpha + 3) * (alpha + 4)); break;
                };
                return result
            },
            centralMoments: function (k, alpha, a, b) {
                const µ1 = this.nonCentralMoments(1, alpha, a, b);
                const µ2 = this.nonCentralMoments(2, alpha, a, b);
                const µ3 = this.nonCentralMoments(3, alpha, a, b);
                const µ4 = this.nonCentralMoments(4, alpha, a, b);

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
                mean: function (alpha, a, b) {
                    const µ1 = dists.power_function.measurements.nonCentralMoments(1, alpha, a, b);
                    return µ1;
                },
                variance: function (alpha, a, b) {
                    const µ1 = dists.power_function.measurements.nonCentralMoments(1, alpha, a, b);
                    const µ2 = dists.power_function.measurements.nonCentralMoments(2, alpha, a, b);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (alpha, a, b) {
                    return Math.sqrt(this.variance(alpha, a, b));
                },
                skewness: function (alpha, a, b) {
                    const central_µ3 = dists.power_function.measurements.centralMoments(3, alpha, a, b);
                    return central_µ3 / (this.standardDeviation(alpha, a, b) ** 3);
                },
                kurtosis: function (alpha, a, b) {
                    const central_µ4 = dists.power_function.measurements.centralMoments(4, alpha, a, b);
                    return central_µ4 / (this.standardDeviation(alpha, a, b) ** 4);
                },
                median: function (alpha, a, b) {
                    return dists.power_function.measurements.ppf(0.5, alpha, a, b);
                },
                mode: function (alpha, a, b) {
                    return Math.max(a, b);
                },
            },
        }
    }
}
console.log(dists.power_function.measurements.stats.mean(2, 10, 11))
console.log(dists.power_function.measurements.stats.variance(2, 10, 11))
console.log(dists.power_function.measurements.stats.standardDeviation(2, 10, 11))
console.log(dists.power_function.measurements.stats.skewness(2, 10, 11))
console.log(dists.power_function.measurements.stats.kurtosis(2, 10, 11))
// console.log(dists.power_function.measurements.stats.median(2, 10, 11))
console.log(dists.power_function.measurements.stats.mode(2, 10, 11))

// mean_value: 10.666666666666666
// variance_value: 0.055555555555557135
// standard_deviation_value: 0.2357022603955192
// skewness_value: -0.5656854249409821
// kurtosis_value: 2.399999999964939
// median_value: 10.707106781186548
// mode_value: 11