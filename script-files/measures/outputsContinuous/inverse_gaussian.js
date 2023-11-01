jStat = require("../node_modules/jstat");

dists = {
    inverse_gaussian: {
        measurements: {
            nonCentralMoments: function (k, mu, lambda) {
                return undefined;
            },
            centralMoments: function (k, mu, lambda) {
                return undefined;
            },
            stats: {
                mean: function (mu, lambda) {
                    return mu;
                },
                variance: function (mu, lambda) {
                    return mu**3/lambda;
                },
                standardDeviation: function (mu, lambda) {
                    return Math.sqrt(this.variance(mu, lambda));
                },
                skewness: function (mu, lambda) {
                    return 3*Math.sqrt(mu/lambda);
                },
                kurtosis: function (mu, lambda) {
                    return 15*(mu/lambda)+3;
                },
                median: function (mu, lambda) {
                    return dists.inverse_gaussian.measurements.ppf(0.5, mu, lambda);
                },
                mode: function (mu, lambda) {
                    return mu*(Math.sqrt(1+9*mu*mu/(4*lambda*lambda))-3*mu/(2*lambda));
                },
            },
        }
    }
}
console.log(dists.inverse_gaussian.measurements.stats.mean(10, 20))
console.log(dists.inverse_gaussian.measurements.stats.variance(10, 20))
console.log(dists.inverse_gaussian.measurements.stats.standardDeviation(10, 20))
console.log(dists.inverse_gaussian.measurements.stats.skewness(10, 20))
console.log(dists.inverse_gaussian.measurements.stats.kurtosis(10, 20))
// console.log(dists.inverse_gaussian.measurements.stats.median(10, 20))
console.log(dists.inverse_gaussian.measurements.stats.mode(10, 20))

// mean_value: 10
// variance_value: 50
// standard_deviation_value: 7.0710678118654755
// skewness_value: 2.121320343559643
// kurtosis_value: 10.5
// median_value: None
// mode_value: 5