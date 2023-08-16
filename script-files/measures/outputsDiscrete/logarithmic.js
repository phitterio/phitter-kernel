jStat = require("../node_modules/jstat");

dists = {
    logarithmic: {
        measurements: {
            nonCentralMoments: function (k, p) {
                return undefined;
            },
            centralMoments: function (k, p) {
                return undefined;
            },
            stats: {
                mean: function (p) {
                    return -p / ((1 - p) * Math.log(1 - p));
                },
                variance: function (p) {
                    return (-p * (p + Math.log(1 - p))) / ((1 - p) ** 2 * Math.log(1 - p) ** 2);
                },
                standardDeviation: function (p) {
                    return Math.sqrt(this.variance(p));
                },
                skewness: function (p) {
                    return (
                        (-(2 * p ** 2 + 3 * p * Math.log(1 - p) + (1 + p) * Math.log(1 - p) ** 2) /
                            (Math.log(1 - p) * (p + Math.log(1 - p)) * Math.sqrt(-p * (p + Math.log(1 - p))))) *
                        Math.log(1 - p)
                    );
                },
                kurtosis: function (p) {
                    return (
                        -(6 * p ** 3 + 12 * p ** 2 * Math.log(1 - p) + p * (4 * p + 7) * Math.log(1 - p) ** 2 + (p ** 2 + 4 * p + 1) * Math.log(1 - p) ** 3) /
                        (p * (p + Math.log(1 - p)) ** 2)
                    );
                },
                median: function (p) {
                    return dists.logarithmic.measurements.ppf(0.5, p);
                },
                mode: function (p) {
                    return 1;
                },
            },
        }
    }
}
console.log(dists.logarithmic.measurements.stats.mean(0.8));
console.log(dists.logarithmic.measurements.stats.variance(0.8));
console.log(dists.logarithmic.measurements.stats.standardDeviation(0.8));
console.log(dists.logarithmic.measurements.stats.skewness(0.8));
console.log(dists.logarithmic.measurements.stats.kurtosis(0.8));
// console.log(dists.logarithmic.measurements.stats.median(0.8))
console.log(dists.logarithmic.measurements.stats.mode(0.8));

// mean_value: 2.485339738238448
// variance_value: 6.249785076725086
// standard_deviation_value: 2.499957014975475
// skewness_value: 3.193127376206349
// kurtosis_value: 15.890903600964736
// median_value: 2
// mode_value: 1
