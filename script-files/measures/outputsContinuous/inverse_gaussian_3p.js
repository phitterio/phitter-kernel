jStat = require("../node_modules/jstat");

dists = {
    inverse_gaussian_3p: {
        measurements: {
            nonCentralMoments: function (k, mu, lambda, loc) {
                return undefined;
            },
            centralMoments: function (k, mu, lambda, loc) {
                return undefined;
            },
            stats: {
                mean: function (mu, lambda, loc) {
                    return mu+loc;
                },
                variance: function (mu, lambda, loc) {
                    return (mu**3/lambda);
                },
                standardDeviation: function (mu, lambda, loc) {
                    return Math.sqrt(this.variance(mu, lambda, loc));
                },
                skewness: function (mu, lambda, loc) {
                    return 3*Math.sqrt(mu/lambda);
                },
                kurtosis: function (mu, lambda, loc) {
                    return 15*(mu/lambda)+3;
                },
                median: function (mu, lambda, loc) {
                    return dists.inverse_gaussian_3p.measurements.ppf(0.5, mu, lambda, loc);
                },
                mode: function (mu, lambda, loc) {
                    return loc+mu*(Math.sqrt(1+9*mu*mu/(4*lambda*lambda))-3*mu/(2*lambda));
                },
            },
        }
    }
}
console.log(dists.inverse_gaussian_3p.measurements.stats.mean(10, 100, 60))
console.log(dists.inverse_gaussian_3p.measurements.stats.variance(10, 100, 60))
console.log(dists.inverse_gaussian_3p.measurements.stats.standardDeviation(10, 100, 60))
console.log(dists.inverse_gaussian_3p.measurements.stats.skewness(10, 100, 60))
console.log(dists.inverse_gaussian_3p.measurements.stats.kurtosis(10, 100, 60))
// console.log(dists.inverse_gaussian_3p.measurements.stats.median(10, 100, 60))
console.log(dists.inverse_gaussian_3p.measurements.stats.mode(10, 100, 60))

// mean_value: 70
// variance_value: 10
// standard_deviation_value: 3.1622776601683795
// skewness_value: 0.9486832980505138
// kurtosis_value: 4.5
// median_value: None
// mode_value: 68.61187420807835