jStat = require("../node_modules/jstat");

function digamma(x) {
    return Math.log(x) - 1 / (2 * x);
}

function polygamma(k, x) {
    function suma(x) {
        let suma = 0;
        for (let i = 0; i < 1000; i++) {
            suma += 1 / Math.pow(x + i, k + 1);
        }
        return suma;
    }
    var resultado = Math.pow(-1, k + 1) * jStat.factorial(k) * suma(x);
    return resultado;
}

dists = {
    generalized_logistic: {
        measurements: {
            nonCentralMoments: function (k, c, loc, scale) {
                return undefined;
            },
            centralMoments: function (k, c, loc, scale) {
                return undefined;
            },
            stats: {
                mean: function (c, loc, scale) {
                    return loc + scale * (0.57721 + digamma(c));
                },
                variance: function (c, loc, scale) {
                    return scale * scale * (Math.PI * Math.PI / 6 + polygamma(1, c));
                },
                standardDeviation: function (c, loc, scale) {
                    return Math.sqrt(this.variance(c, loc, scale));
                },
                skewness: function (c, loc, scale) {
                    return (2.40411380631918 + polygamma(2, c)) / ((Math.PI * Math.PI / 6 + polygamma(1, c)) ** 1.5);
                },
                kurtosis: function (c, loc, scale) {
                    return 3 + (6.49393940226682 + polygamma(3, c)) / ((Math.PI * Math.PI / 6 + polygamma(1, c)) ** 2);
                },
                median: function (c, loc, scale) {
                    return dists.generalized_logistic.measurements.ppf(0.5, c, loc, scale);
                },
                mode: function (c, loc, scale) {
                    return loc + scale * Math.log(c);
                },
            },
        }
    }
}
console.log(dists.generalized_logistic.measurements.stats.mean(5, 10, 34))
console.log(dists.generalized_logistic.measurements.stats.variance(5, 10, 34))
console.log(dists.generalized_logistic.measurements.stats.standardDeviation(5, 10, 34))
console.log(dists.generalized_logistic.measurements.stats.skewness(5, 10, 34))
console.log(dists.generalized_logistic.measurements.stats.kurtosis(5, 10, 34))
// console.log(dists.generalized_logistic.measurements.stats.median(5, 10, 34))
console.log(dists.generalized_logistic.measurements.stats.mode(5, 10, 34))

// mean_value: 80.9460290227594
// variance_value: 2156.242296899568
// standard_deviation_value: 46.43535610824545
// skewness_value: 0.9245731210131928
// kurtosis_value: 4.872662204876139
// median_value: 74.79840659673239
// mode_value: 64.72088902275941