jStat = require("../node_modules/jstat");

dists = {
    negative_binomial: {
        measurements: {
            nonCentralMoments: function (k, r, p) {
                return undefined;
            },
            centralMoments: function (k, r, p) {
                return undefined;
            },
            stats: {
                mean: function (r, p) {
                    return (r * (1 - p)) / p;
                },
                variance: function (r, p) {
                    return (r * (1 - p)) / (p * p);
                },
                standardDeviation: function (r, p) {
                    return Math.sqrt(this.variance(r, p));
                },
                skewness: function (r, p) {
                    return (2 - p) / Math.sqrt(r * (1 - p));
                },
                kurtosis: function (r, p) {
                    return 6 / r + (p * p) / (r * (1 - p)) + 3;
                },
                median: function (r, p) {
                    return dists.negative_binomial.measurements.ppf(0.5, r, p);
                },
                mode: function (r, p) {
                    return Math.floor(((r - 1) * (1 - p)) / p, 0);
                },
            },
        }
    }
}
console.log(dists.negative_binomial.measurements.stats.mean(10, 0.7));
console.log(dists.negative_binomial.measurements.stats.variance(10, 0.7));
console.log(dists.negative_binomial.measurements.stats.standardDeviation(10, 0.7));
console.log(dists.negative_binomial.measurements.stats.skewness(10, 0.7));
console.log(dists.negative_binomial.measurements.stats.kurtosis(10, 0.7));
// console.log(dists.negative_binomial.measurements.stats.median(10, 0.7))
console.log(dists.negative_binomial.measurements.stats.mode(10, 0.7));

// mean_value: 4.2857142857142865
// variance_value: 6.122448979591838
// standard_deviation_value: 2.474358296526968
// skewness_value: 0.7505553499465135
// kurtosis_value: 3.763333333333333
// median_value: 4
// mode_value: 3
