jStat = require("../node_modules/jstat");

dists = {
    beta: {
        measurements: {
            nonCentralMoments: function (k, alpha, beta, A, B) {
                return undefined;
            },
            centralMoments: function (k, alpha, beta, A, B) {
                return undefined;
            },
            stats: {
                mean: function (alpha, beta, A, B) {
                    return A + (alpha / (alpha + beta)) * (B - A);
                },
                variance: function (alpha, beta, A, B) {
                    return (alpha * beta / ((alpha + beta + 1) * (alpha + beta) ** 2)) * ((B - A) ** 2);
                },
                standardDeviation: function (alpha, beta, A, B) {
                    return Math.sqrt(this.variance(alpha, beta, A, B));
                },
                skewness: function (alpha, beta, A, B) {
                    return 2 * ((beta - alpha) / (alpha + beta + 2)) * Math.sqrt((alpha + beta + 1) / (alpha * beta));
                },
                kurtosis: function (alpha, beta, A, B) {
                    return 3 + 6 * ((alpha + beta + 1) * (alpha - beta) ** 2 - alpha * beta * (alpha + beta + 2)) / (alpha * beta * (alpha + beta + 2) * (alpha + beta + 3));
                },
                median: function (alpha, beta, A, B) {
                    return dists.beta.measurements.ppf(0.5, alpha, beta, A, B);
                },
                mode: function (alpha, beta, A, B) {
                    return A + ((alpha - 1) / (alpha + beta - 2)) * (B - A);
                },
            },
        }
    }
}
console.log(dists.beta.measurements.stats.mean(2, 3, 300, 1000))
console.log(dists.beta.measurements.stats.variance(2, 3, 300, 1000))
console.log(dists.beta.measurements.stats.standardDeviation(2, 3, 300, 1000))
console.log(dists.beta.measurements.stats.skewness(2, 3, 300, 1000))
console.log(dists.beta.measurements.stats.kurtosis(2, 3, 300, 1000))
// console.log(dists.beta.measurements.stats.median(2, 3, 300, 1000))
console.log(dists.beta.measurements.stats.mode(2, 3, 300, 1000))

// mean_value: 580
// variance_value: 19600
// standard_deviation_value: 140
// skewness_value: 0.2857142857142857
// kurtosis_value: 2.357142857142857
// median_value: 570.0092976926726
// mode_value: 533.3333333333333