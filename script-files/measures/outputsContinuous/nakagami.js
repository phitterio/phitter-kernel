jStat = require("../node_modules/jstat");

dists = {
    nakagami: {
        measurements: {
            nonCentralMoments: function (k, m, omega) {
                return undefined;
            },
            centralMoments: function (k, m, omega) {
                return undefined;
            },
            stats: {
                mean: function (m, omega) {
                    return (jStat.gammafn(m + 0.5) / jStat.gammafn(m)) * Math.sqrt(omega / m);
                },
                variance: function (m, omega) {
                    return omega * (1 - (1 / m) * (jStat.gammafn(m + 0.5) / jStat.gammafn(m)) ** 2);
                },
                standardDeviation: function (m, omega) {
                    return Math.sqrt(this.variance(m, omega));
                },
                skewness: function (m, omega) {
                    return ((jStat.gammafn(m + 0.5) / jStat.gammafn(m)) * Math.sqrt(1 / m)) * (1 - 4 * m * (1 - ((jStat.gammafn(m + 0.5) / jStat.gammafn(m)) * Math.sqrt(1 / m)) ** 2)) / (2 * m * (1 - ((jStat.gammafn(m + 0.5) / jStat.gammafn(m)) * Math.sqrt(1 / m)) ** 2) ** 1.5);
                },
                kurtosis: function (m, omega) {
                    return 3 + (-6 * ((jStat.gammafn(m + 0.5) / jStat.gammafn(m)) * Math.sqrt(1 / m)) ** 4 * m + (8 * m - 2) * ((jStat.gammafn(m + 0.5) / jStat.gammafn(m)) * Math.sqrt(1 / m)) ** 2 - 2 * m + 1) / (m * (1 - ((jStat.gammafn(m + 0.5) / jStat.gammafn(m)) * Math.sqrt(1 / m)) ** 2) ** 2);
                },
                median: function (m, omega) {
                    return dists.nakagami.measurements.ppf(0.5, m, omega);
                },
                mode: function (m, omega) {
                    return (Math.sqrt(2) / 2) * Math.sqrt(omega * (2 * m - 1) / (m));
                },
            },
        }
    }
}
console.log(dists.nakagami.measurements.stats.mean(3, 19))
console.log(dists.nakagami.measurements.stats.variance(3, 19))
console.log(dists.nakagami.measurements.stats.standardDeviation(3, 19))
console.log(dists.nakagami.measurements.stats.skewness(3, 19))
console.log(dists.nakagami.measurements.stats.kurtosis(3, 19))
// console.log(dists.nakagami.measurements.stats.median(3, 19))
console.log(dists.nakagami.measurements.stats.mode(3, 19))

// mean_value: 4.181791599529597
// variance_value: 1.512619018103691
// standard_deviation_value: 1.229885774413092
// skewness_value: 0.3179108769305244
// kurtosis_value: 3.025111348684794
// median_value: 4.1153025794283
// mode_value: 3.9791121287711073