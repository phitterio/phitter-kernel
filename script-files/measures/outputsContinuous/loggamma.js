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
    loggamma: {
        measurements: {
            nonCentralMoments: function (k, c, miu, sigma) {
                return undefined;
            },
            centralMoments: function (k, c, miu, sigma) {
                return undefined;
            },
            stats: {
                mean: function (c, miu, sigma) {
                    return digamma(c) * sigma + miu;
                },
                variance: function (c, miu, sigma) {
                    return polygamma(1, c) * sigma * sigma;
                },
                standardDeviation: function (c, miu, sigma) {
                    return Math.sqrt(this.variance(c, miu, sigma));
                },
                skewness: function (c, miu, sigma) {
                    return polygamma(2, c) / (polygamma(1, c) ** 1.5);
                },
                kurtosis: function (c, miu, sigma) {
                    return polygamma(3, c) / (polygamma(1, c) ** 2) + 3;
                },
                median: function (c, miu, sigma) {
                    return dists.loggamma.measurements.ppf(0.5, c, miu, sigma);
                },
                mode: function (c, miu, sigma) {
                    return miu + sigma * Math.log(c);
                },
            },
        }
    }
}
console.log(dists.loggamma.measurements.stats.mean(2, 10, 4))
console.log(dists.loggamma.measurements.stats.variance(2, 10, 4))
console.log(dists.loggamma.measurements.stats.standardDeviation(2, 10, 4))
console.log(dists.loggamma.measurements.stats.skewness(2, 10, 4))
console.log(dists.loggamma.measurements.stats.kurtosis(2, 10, 4))
// console.log(dists.loggamma.measurements.stats.median(2, 10, 4))
console.log(dists.loggamma.measurements.stats.mode(2, 10, 4))

// mean_value: 11.772588722239782
// variance_value: 10.302969034952858
// standard_deviation_value: 3.209823832385955
// skewness_value: -0.7820580644763452
// kurtosis_value: 4.191211414406338
// median_value: 12.071237498079809
// mode_value: 12.772588722239782