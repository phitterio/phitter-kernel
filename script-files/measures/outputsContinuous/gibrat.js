jStat = require("../node_modules/jstat");

dists = {
    gibrat: {
        measurements: {
            nonCentralMoments: function (k, loc, scale) {
                return undefined;
            },
            centralMoments: function (k, loc, scale) {
                return undefined;
            },
            stats: {
                mean: function (loc, scale) {
                    return loc + scale * Math.sqrt(Math.exp(1));
                },
                variance: function (loc, scale) {
                    return Math.exp(1) * (Math.exp(1) - 1) * scale * scale;
                },
                standardDeviation: function (loc, scale) {
                    return Math.sqrt(this.variance(loc, scale));
                },
                skewness: function (loc, scale) {
                    return (2 + Math.exp(1)) * Math.sqrt(Math.exp(1) - 1);
                },
                kurtosis: function (loc, scale) {
                    return Math.exp(1) ** 4 + 2 * Math.exp(1) ** 3 + 3 * Math.exp(1) ** 2 - 6;
                },
                median: function (loc, scale) {
                    return dists.gibrat.measurements.ppf(0.5, loc, scale);
                },
                mode: function (loc, scale) {
                    return (1 / Math.exp(1)) * scale + loc;
                },
            },
        }
    }
}
console.log(dists.gibrat.measurements.stats.mean(20, 100))
console.log(dists.gibrat.measurements.stats.variance(20, 100))
console.log(dists.gibrat.measurements.stats.standardDeviation(20, 100))
console.log(dists.gibrat.measurements.stats.skewness(20, 100))
console.log(dists.gibrat.measurements.stats.kurtosis(20, 100))
// console.log(dists.gibrat.measurements.stats.median(20, 100))
console.log(dists.gibrat.measurements.stats.mode(20, 100))

// mean_value: 184.87212707001282
// variance_value: 46707.742704716045
// standard_deviation_value: 216.11974158950878
// skewness_value: 6.184877138632554
// kurtosis_value: 110.9363921763115
// median_value: 120
// mode_value: 56.787944117144235