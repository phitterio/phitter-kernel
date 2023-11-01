jStat = require("../node_modules/jstat");

dists = {
    pert: {
        measurements: {
            nonCentralMoments: function (k, a, b, c) {
                return undefined;
            },
            centralMoments: function (k, a, b, c) {
                return undefined;
            },
            stats: {
                mean: function (a, b, c) {
                    return (a + 4 * b + c) / 6;
                },
                variance: function (a, b, c) {
                    return ((this.mean(a, b, c) - a) * (c - this.mean(a, b, c))) / 7;
                },
                standardDeviation: function (a, b, c) {
                    return Math.sqrt(this.variance(a, b, c));
                },
                skewness: function (a, b, c) {
                    const alpha1 = (4 * b + c - 5 * a) / (c - a);
                    const alpha2 = (5 * c - a - 4 * b) / (c - a);
                    return 2 * (alpha2 - alpha1) * Math.sqrt(alpha1 + alpha2 + 1) / ((alpha1 + alpha2 + 2) * (Math.sqrt(alpha1 * alpha2)));
                },
                kurtosis: function (a, b, c) {
                    const alpha1 = (4 * b + c - 5 * a) / (c - a);
                    const alpha2 = (5 * c - a - 4 * b) / (c - a);
                    return 6 * (((alpha2 - alpha1) ** 2) * (alpha1 + alpha2 + 1) - alpha1 * alpha2 * (alpha1 + alpha2 + 2)) / (alpha1 * alpha2 * (alpha1 + alpha2 + 2) * (alpha1 + alpha2 + 3)) + 3;
                },
                median: function (a, b, c) {
                    return dists.pert.measurements.ppf(0.5, a, b, c);
                },
                mode: function (a, b, c) {
                    return b;
                },
            },
        }
    }
}
console.log(dists.pert.measurements.stats.mean(100, 500, 1000))
console.log(dists.pert.measurements.stats.variance(100, 500, 1000))
console.log(dists.pert.measurements.stats.standardDeviation(100, 500, 1000))
console.log(dists.pert.measurements.stats.skewness(100, 500, 1000))
console.log(dists.pert.measurements.stats.kurtosis(100, 500, 1000))
// console.log(dists.pert.measurements.stats.median(100, 500, 1000))
console.log(dists.pert.measurements.stats.mode(100, 500, 1000))

// mean_value: 516.6666666666666
// variance_value: 28769.84126984127
// standard_deviation_value: 169.61674819970247
// skewness_value: 0.09826073688810355
// kurtosis_value: 2.3462068965517244
// median_value: 512.7323165104102
// mode_value: 500