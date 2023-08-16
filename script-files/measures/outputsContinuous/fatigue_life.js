jStat = require("../node_modules/jstat");

dists = {
    fatigue_life: {
        measurements: {
            nonCentralMoments: function (k, gamma, loc, scale) {
                return undefined;
            },
            centralMoments: function (k, gamma, loc, scale) {
                return undefined;
            },
            stats: {
                mean: function (gamma, loc, scale) {
                    return loc + scale * (1 + (gamma ** 2) / 2);
                },
                variance: function (gamma, loc, scale) {
                    return scale ** 2 * gamma ** 2 * (1 + 5 * gamma ** 2 / 4);
                },
                standardDeviation: function (gamma, loc, scale) {
                    return Math.sqrt(this.variance(gamma, loc, scale));
                },
                skewness: function (gamma, loc, scale) {
                    return 4 * gamma ** 2 * (11 * gamma ** 2 + 6) / ((5 * gamma ** 2 + 4) * Math.sqrt(gamma ** 2 * (5 * gamma ** 2 + 4)));
                },
                kurtosis: function (gamma, loc, scale) {
                    return 3 + (6 * gamma * gamma * (93 * gamma * gamma + 40)) / ((5 * gamma ** 2 + 4) ** 2);
                },
                median: function (gamma, loc, scale) {
                    return dists.fatigue_life.measurements.ppf(0.5, gamma, loc, scale);
                },
                mode: function (gamma, loc, scale) {
                    return undefined;
                },
            },
        }
    }
}
console.log(dists.fatigue_life.measurements.stats.mean(3, 15, 5))
console.log(dists.fatigue_life.measurements.stats.variance(3, 15, 5))
console.log(dists.fatigue_life.measurements.stats.standardDeviation(3, 15, 5))
console.log(dists.fatigue_life.measurements.stats.skewness(3, 15, 5))
console.log(dists.fatigue_life.measurements.stats.kurtosis(3, 15, 5))
// console.log(dists.fatigue_life.measurements.stats.median(3, 15, 5))
console.log(dists.fatigue_life.measurements.stats.mode(3, 15, 5))

// mean_value: 42.5
// variance_value: 2756.25
// standard_deviation_value: 52.5
// skewness_value: 3.673469387755102
// kurtosis_value: 22.724281549354437
// median_value: 20
// mode_value: complicated