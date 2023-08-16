jStat = require("../node_modules/jstat");

dists = {
    geometric: {
        measurements: {
            nonCentralMoments: function (k, p) {
                return undefined;
            },
            centralMoments: function (k, p) {
                return undefined;
            },
            stats: {
                mean: function (p) {
                    return 1 / p;
                },
                variance: function (p) {
                    return (1 - p) / (p * p);
                },
                standardDeviation: function (p) {
                    return Math.sqrt(this.variance(p));
                },
                skewness: function (p) {
                    return (2 - p) / Math.sqrt(1 - p);
                },
                kurtosis: function (p) {
                    return 3 + 6 + (p * p) / (1 - p);
                },
                median: function (p) {
                    return dists.geometric.measurements.ppf(0.5, p);
                },
                mode: function (p) {
                    return 1.0;
                },
            },
        }
    }
}
console.log(dists.geometric.measurements.stats.mean(0.2));
console.log(dists.geometric.measurements.stats.variance(0.2));
console.log(dists.geometric.measurements.stats.standardDeviation(0.2));
console.log(dists.geometric.measurements.stats.skewness(0.2));
console.log(dists.geometric.measurements.stats.kurtosis(0.2));
// console.log(dists.geometric.measurements.stats.median(0.2))
console.log(dists.geometric.measurements.stats.mode(0.2));

// mean_value: 5
// variance_value: 20
// standard_deviation_value: 4.472135955
// skewness_value: 2.01246118
// kurtosis_value: 9.05
// median_value: 4
// mode_value: 1.0
