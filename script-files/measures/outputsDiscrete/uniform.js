jStat = require("jstat");

dists = {
    uniform: {
        measurements: {
            nonCentralMoments: function (k, a, b) {
                return undefined;
            },
            centralMoments: function (k, a, b) {
                return undefined;
            },
            stats: {
                mean: function (a, b) {
                    return (a + b) / 2;
                },
                variance: function (a, b) {
                    return ((b - a + 1) * (b - a + 1) - 1) / 12;
                },
                standardDeviation: function (a, b) {
                    return Math.sqrt(this.variance(a, b));
                },
                skewness: function (a, b) {
                    return 0;
                },
                kurtosis: function (a, b) {
                    return ((-6 / 5) * ((b - a + 1) * (b - a + 1) + 1)) / ((b - a + 1) * (b - a + 1) - 1) + 3;
                },
                median: function (a, b) {
                    return dists.uniform.measurements.ppf(0.5, a, b);
                },
                mode: function (a, b) {
                    return undefined;
                },
            },
        }
    }
}
console.log(dists.uniform.measurements.stats.mean(3, 10));
console.log(dists.uniform.measurements.stats.variance(3, 10));
console.log(dists.uniform.measurements.stats.standardDeviation(3, 10));
console.log(dists.uniform.measurements.stats.skewness(3, 10));
console.log(dists.uniform.measurements.stats.kurtosis(3, 10));
// console.log(dists.uniform.measurements.stats.median(3, 10))
console.log(dists.uniform.measurements.stats.mode(3, 10));

// mean_value: 6.5
// variance_value: 5.25
// standard_deviation_value: 2.29128784747792
// skewness_value: 0
// kurtosis_value: 1.7619047619047619
// median_value: None
// mode_value: None
