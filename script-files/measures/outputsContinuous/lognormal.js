jStat = require("../node_modules/jstat");

dists = {
    lognormal: {
        measurements: {
            nonCentralMoments: function (k, mu, sigma) {
                return undefined;
            },
            centralMoments: function (k, mu, sigma) {
                return undefined;
            },
            stats: {
                mean: function (mu, sigma) {
                    return Math.exp(mu + (sigma ** 2) / 2);
                },
                variance: function (mu, sigma) {
                    return (Math.exp(sigma ** 2) - 1) * Math.exp(2 * mu + sigma ** 2);
                },
                standardDeviation: function (mu, sigma) {
                    return Math.sqrt(this.variance(mu, sigma));
                },
                skewness: function (mu, sigma) {
                    return (Math.exp(sigma * sigma) + 2) * Math.sqrt(Math.exp(sigma * sigma) - 1);
                },
                kurtosis: function (mu, sigma) {
                    return Math.exp(4 * sigma * sigma) + 2 * Math.exp(3 * sigma * sigma) + 3 * Math.exp(2 * sigma * sigma) - 3;
                },
                median: function (mu, sigma) {
                    return dists.lognormal.measurements.ppf(0.5, mu, sigma);
                },
                mode: function (mu, sigma) {
                    return Math.exp(mu - sigma * sigma);
                },
            },
        }
    }
}
console.log(dists.lognormal.measurements.stats.mean(3, 0.3))
console.log(dists.lognormal.measurements.stats.variance(3, 0.3))
console.log(dists.lognormal.measurements.stats.standardDeviation(3, 0.3))
console.log(dists.lognormal.measurements.stats.skewness(3, 0.3))
console.log(dists.lognormal.measurements.stats.kurtosis(3, 0.3))
// console.log(dists.lognormal.measurements.stats.median(3, 0.3))
console.log(dists.lognormal.measurements.stats.mode(3, 0.3))

// mean_value: 21.01003120287951
// variance_value: 41.57054520681497
// standard_deviation_value: 6.4475224083996
// skewness_value: 0.949534907256536
// kurtosis_value: 4.644910405392265
// median_value: 20.085536923187668
// mode_value: 18.356798567017925