jStat = require("../node_modules/jstat");

dists = {
    johnson_su: {
        measurements: {
            nonCentralMoments: function (k, xi, lambda, gamma, delta) {
                return undefined;
            },
            centralMoments: function (k, xi, lambda, gamma, delta) {
                return undefined;
            },
            stats: {
                mean: function (xi, lambda, gamma, delta) {
                    return xi - lambda * Math.exp((delta ** -2) / 2) * Math.sinh(gamma / delta);
                },
                variance: function (xi, lambda, gamma, delta) {
                    return ((lambda ** 2) / 2) * (Math.exp(delta ** -2) - 1) * (Math.exp(delta ** -2) * Math.cosh(2 * gamma / delta) + 1);
                },
                standardDeviation: function (xi, lambda, gamma, delta) {
                    return Math.sqrt(this.variance(xi, lambda, gamma, delta));
                },
                skewness: function (xi, lambda, gamma, delta) {
                    return -((lambda ** 3) * Math.sqrt(Math.exp(delta ** -2)) * ((Math.exp(delta ** -2) - 1) ** 2) * (Math.exp(delta ** -2) * (Math.exp(delta ** -2) + 2) * Math.sinh(3 * (gamma / delta)) + 3 * Math.sinh((gamma / delta)))) / (4 * this.standardDeviation(xi, lambda, gamma, delta) ** 3);
                },
                kurtosis: function (xi, lambda, gamma, delta) {
                    return ((lambda ** 4) * ((Math.exp(delta ** -2) - 1) ** 2) * ((Math.exp(delta ** -2) ** 2) * (Math.exp(delta ** -2) ** 4 + 2 * Math.exp(delta ** -2) ** 3 + 3 * Math.exp(delta ** -2) ** 2 - 3) * Math.cosh(4 * (gamma / delta)) + (4 * Math.exp(delta ** -2) ** 2) * (Math.exp(delta ** -2) + 2) * Math.cosh(2 * (gamma / delta)) + 3 * (2 * Math.exp(delta ** -2) + 1))) / (8 * this.standardDeviation(xi, lambda, gamma, delta) ** 4);
                },
                median: function (xi, lambda, gamma, delta) {
                    return dists.johnson_su.measurements.ppf(0.5, xi, lambda, gamma, delta);
                },
                mode: function (xi, lambda, gamma, delta) {
                    return undefined;
                },
            },
        }
    }
}
console.log(dists.johnson_su.measurements.stats.mean(100, 65, 8, 15))
console.log(dists.johnson_su.measurements.stats.variance(100, 65, 8, 15))
console.log(dists.johnson_su.measurements.stats.standardDeviation(100, 65, 8, 15))
console.log(dists.johnson_su.measurements.stats.skewness(100, 65, 8, 15))
console.log(dists.johnson_su.measurements.stats.kurtosis(100, 65, 8, 15))
// console.log(dists.johnson_su.measurements.stats.median(100, 65, 8, 15))
console.log(dists.johnson_su.measurements.stats.mode(100, 65, 8, 15))

// mean_value: 63.58551277411177
// variance_value: 24.767997731072235
// standard_deviation_value: 4.976745696845704
// skewness_value: -0.09797660570383238
// kurtosis_value: 3.0307067587072023
// median_value: 63.66634401108656
// mode_value: None