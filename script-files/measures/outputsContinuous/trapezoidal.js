jStat = require("../node_modules/jstat");

dists = {
    trapezoidal: {
        measurements: {
            nonCentralMoments: function (k, a, b, c, d) {
                return (2 / ((d + c - b - a))) * (1 / ((k + 1) * (k + 2))) * ((d ** (k + 2) - c ** (k + 2)) / (d - c) - (b ** (k + 2) - a ** (k + 2)) / (b - a))
            },
            centralMoments: function (k, a, b, c, d) {
                const µ1 = this.nonCentralMoments(1, a, b, c, d);
                const µ2 = this.nonCentralMoments(2, a, b, c, d);
                const µ3 = this.nonCentralMoments(3, a, b, c, d);
                const µ4 = this.nonCentralMoments(4, a, b, c, d);

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
                mean: function (a, b, c, d) {
                    const µ1 = dists.trapezoidal.measurements.nonCentralMoments(1, a, b, c, d);
                    return µ1;
                },
                variance: function (a, b, c, d) {
                    const µ1 = dists.trapezoidal.measurements.nonCentralMoments(1, a, b, c, d);
                    const µ2 = dists.trapezoidal.measurements.nonCentralMoments(2, a, b, c, d);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (a, b, c, d) {
                    return Math.sqrt(this.variance(a, b, c, d));
                },
                skewness: function (a, b, c, d) {
                    const central_µ3 = dists.trapezoidal.measurements.centralMoments(3, a, b, c, d);
                    return central_µ3 / (this.standardDeviation(a, b, c, d) ** 3);
                },
                kurtosis: function (a, b, c, d) {
                    const central_µ4 = dists.trapezoidal.measurements.centralMoments(4, a, b, c, d);
                    return central_µ4 / (this.standardDeviation(a, b, c, d) ** 4);
                },
                median: function (a, b, c, d) {
                    return dists.trapezoidal.measurements.ppf(0.5, a, b, c, d);
                },
                mode: function (a, b, c, d) {
                    return undefined;
                },
            },
        }
    }
}
console.log(dists.trapezoidal.measurements.stats.mean(100, 200, 700, 1000))
console.log(dists.trapezoidal.measurements.stats.variance(100, 200, 700, 1000))
console.log(dists.trapezoidal.measurements.stats.standardDeviation(100, 200, 700, 1000))
console.log(dists.trapezoidal.measurements.stats.skewness(100, 200, 700, 1000))
console.log(dists.trapezoidal.measurements.stats.kurtosis(100, 200, 700, 1000))
// console.log(dists.trapezoidal.measurements.stats.median(100, 200, 700, 1000))
console.log(dists.trapezoidal.measurements.stats.mode(100, 200, 700, 1000))

// mean_value: 504.76190476190476
// variance_value: 44977.32426303852
// standard_deviation_value: 212.0785803966033
// skewness_value: 0.11983497302112443
// kurtosis_value: 1.999763336154445
// median_value: 500
// mode_value: -