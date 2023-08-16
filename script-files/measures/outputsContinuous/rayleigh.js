jStat = require("../node_modules/jstat");

dists = {
    rayleigh: {
        measurements: {
            nonCentralMoments: function (k, gamma, sigma) {
                return undefined;
            },
            centralMoments: function (k, gamma, sigma) {
                return undefined;
            },
            stats: {
                mean: function (gamma, sigma) {
                    return sigma * Math.sqrt(Math.PI / 2) + gamma;
                },
                variance: function (gamma, sigma) {
                    return sigma * sigma * (2 - Math.PI / 2);
                },
                standardDeviation: function (gamma, sigma) {
                    return Math.sqrt(this.variance(gamma, sigma));
                },
                skewness: function (gamma, sigma) {
                    return 0.6311;
                },
                kurtosis: function (gamma, sigma) {
                    return (24 * Math.PI - 6 * Math.PI * Math.PI - 16) / ((4 - Math.PI) * (4 - Math.PI)) + 3;
                },
                median: function (gamma, sigma) {
                    return dists.rayleigh.measurements.ppf(0.5, gamma, sigma);
                },
                mode: function (gamma, sigma) {
                    return gamma + sigma;
                },
            },
        }
    }
}
console.log(dists.rayleigh.measurements.stats.mean(10, 2))
console.log(dists.rayleigh.measurements.stats.variance(10, 2))
console.log(dists.rayleigh.measurements.stats.standardDeviation(10, 2))
console.log(dists.rayleigh.measurements.stats.skewness(10, 2))
console.log(dists.rayleigh.measurements.stats.kurtosis(10, 2))
// console.log(dists.rayleigh.measurements.stats.median(10, 2))
console.log(dists.rayleigh.measurements.stats.mode(10, 2))

// mean_value: 12.506628274631
// variance_value: 1.7168146928204138
// standard_deviation_value: 1.3102727551240672
// skewness_value: 0.6311
// kurtosis_value: 3.245089300687639
// median_value: 12.35482004503095
// mode_value: 12