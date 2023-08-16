jStat = require("../node_modules/jstat");

dists = {
    maxwell: {
        measurements: {
            nonCentralMoments: function (k, alpha, loc) {
                return undefined;
            },
            centralMoments: function (k, alpha, loc) {
                return undefined;
            },
            stats: {
                mean: function (alpha, loc) {
                    return 2 * Math.sqrt(2 / Math.PI) * alpha + loc;
                },
                variance: function (alpha, loc) {
                    return alpha * alpha * (3 * Math.PI - 8) / Math.PI;
                },
                standardDeviation: function (alpha, loc) {
                    return Math.sqrt(this.variance(alpha, loc));
                },
                skewness: function (alpha, loc) {
                    return 2 * Math.sqrt(2) * (16 - 5 * Math.PI) / ((3 * Math.PI - 8) ** 1.5);
                },
                kurtosis: function (alpha, loc) {
                    return 4 * (-96 + 40 * Math.PI - 3 * Math.PI * Math.PI) / ((3 * Math.PI - 8) ** 2) + 3;
                },
                median: function (alpha, loc) {
                    return dists.maxwell.measurements.ppf(0.5, alpha, loc);
                },
                mode: function (alpha, loc) {
                    return Math.sqrt(2) * alpha + loc;
                },
            },
        }
    }
}
console.log(dists.maxwell.measurements.stats.mean(20, 100))
console.log(dists.maxwell.measurements.stats.variance(20, 100))
console.log(dists.maxwell.measurements.stats.standardDeviation(20, 100))
console.log(dists.maxwell.measurements.stats.skewness(20, 100))
console.log(dists.maxwell.measurements.stats.kurtosis(20, 100))
// console.log(dists.maxwell.measurements.stats.median(20, 100))
console.log(dists.maxwell.measurements.stats.mode(20, 100))

// mean_value: 131.91538243211463
// variance_value: 181.40836421186978
// standard_deviation_value: 13.468792232857027
// skewness_value: 0.485692828049592
// kurtosis_value: 3.108163842816288
// median_value: 130.76344508910105
// mode_value: 128.2842712474619