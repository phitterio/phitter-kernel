jStat = require("../node_modules/jstat");

dists = {
    bradford: {
        measurements: {
            nonCentralMoments: function (k, c, min, max) {
                return undefined;
            },
            centralMoments: function (k, c, min, max) {
                return undefined;
            },
            stats: {
                mean: function (c, min, max) {
                    return (c * (max - min) + Math.log(1 + c) * (min * (c + 1) - max)) / (Math.log(1 + c) * c);
                },
                variance: function (c, min, max) {
                    return (max - min) ** 2 * ((c + 2) * Math.log(1 + c) - 2 * c) / (2 * c * Math.log(1 + c) ** 2);
                },
                standardDeviation: function (c, min, max) {
                    return Math.sqrt(this.variance(c, min, max));
                },
                skewness: function (c, min, max) {
                    return (Math.sqrt(2) * (12 * c * c - 9 * Math.log(1 + c) * c * (c + 2) + 2 * Math.log(1 + c) * Math.log(1 + c) * (c * (c + 3) + 3))) / (Math.sqrt(c * (c * (Math.log(1 + c) - 2) + 2 * Math.log(1 + c))) * (3 * c * (Math.log(1 + c) - 2) + 6 * Math.log(1 + c)));
                },
                kurtosis: function (c, min, max) {
                    return (c ** 3 * (Math.log(1 + c) - 3) * (Math.log(1 + c) * (3 * Math.log(1 + c) - 16) + 24) + 12 * Math.log(1 + c) * c * c * (Math.log(1 + c) - 4) * (Math.log(1 + c) - 3) + 6 * c * Math.log(1 + c) ** 2 * (3 * Math.log(1 + c) - 14) + 12 * Math.log(1 + c) ** 3) / (3 * c * (c * (Math.log(1 + c) - 2) + 2 * Math.log(1 + c)) ** 2) + 3;
                },
                median: function (c, min, max) {
                    return dists.bradford.measurements.ppf(0.5, c, min, max);
                },
                mode: function (c, min, max) {
                    return min;
                },
            },
        }
    }
}
console.log(dists.bradford.measurements.stats.mean(5, 20, 50))
console.log(dists.bradford.measurements.stats.variance(5, 20, 50))
console.log(dists.bradford.measurements.stats.standardDeviation(5, 20, 50))
console.log(dists.bradford.measurements.stats.skewness(5, 20, 50))
console.log(dists.bradford.measurements.stats.kurtosis(5, 20, 50))
// console.log(dists.bradford.measurements.stats.median(5, 20, 50))
console.log(dists.bradford.measurements.stats.mode(5, 20, 50))

// mean_value: 30.743318796537416
// variance_value: 71.27097040480258
// standard_deviation_value: 8.44221359625558
// skewness_value: 0.6079830478071038
// kurtosis_value: 2.2053933128169976
// median_value: 28.696938456699066
// mode_value: 20