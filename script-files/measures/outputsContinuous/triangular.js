jStat = require("../node_modules/jstat");

dists = {
    triangular: {
        measurements: {
            nonCentralMoments: function (k, a, b, c) {
                return undefined;
            },
            centralMoments: function (k, a, b, c) {
                return undefined;
            },
            stats: {
                mean: function (a, b, c) {
                    return (a + b + c) / 3;
                },
                variance: function (a, b, c) {
                    return (a ** 2 + b ** 2 + c ** 2 - a * b - a * c - b * c) / 18;
                },
                standardDeviation: function (a, b, c) {
                    return Math.sqrt(this.variance(a, b, c));
                },
                skewness: function (a, b, c) {
                    return Math.sqrt(2) * (a + b - 2 * c) * (2 * a - b - c) * (a - 2 * b + c) / (5 * (a ** 2 + b ** 2 + c ** 2 - a * b - a * c - b * c) ** (3 / 2));
                },
                kurtosis: function (a, b, c) {
                    return 3 - 3 / 5;
                },
                median: function (a, b, c) {
                    return dists.triangular.measurements.ppf(0.5, a, b, c);
                },
                mode: function (a, b, c) {
                    return c;
                },
            },
        }
    }
}
console.log(dists.triangular.measurements.stats.mean(100, 1000, 200))
console.log(dists.triangular.measurements.stats.variance(100, 1000, 200))
console.log(dists.triangular.measurements.stats.standardDeviation(100, 1000, 200))
console.log(dists.triangular.measurements.stats.skewness(100, 1000, 200))
console.log(dists.triangular.measurements.stats.kurtosis(100, 1000, 200))
// console.log(dists.triangular.measurements.stats.median(100, 1000, 200))
console.log(dists.triangular.measurements.stats.mode(100, 1000, 200))

// mean_value: 433.3333333333333
// variance_value: 40555.555555555555
// standard_deviation_value: 201.38409955990954
// skewness_value: 0.5396443876366115
// kurtosis_value: 2.4
// median_value: 400
// mode_value: 200