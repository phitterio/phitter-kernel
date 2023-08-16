jStat = require("../node_modules/jstat");

dists = {
    hypergeometric: {
        measurements: {
            nonCentralMoments: function (k, N, K, n) {
                return undefined;
            },
            centralMoments: function (k, N, K, n) {
                return undefined;
            },
            stats: {
                mean: function (N, K, n) {
                    return n * K / N;
                },
                variance: function (N, K, n) {
                    return (n * K / N) * ((N - K) / N) * ((N - n) / (N - 1));
                },
                standardDeviation: function (N, K, n) {
                    return Math.sqrt(this.variance(N, K, n));
                },
                skewness: function (N, K, n) {
                    return ((N - 2 * K) * Math.sqrt(N - 1) * (N - 2 * n)) / (Math.sqrt(n * K * (N - K) * (N - n)) * (N - 2));
                },
                kurtosis: function (N, K, n) {
                    return 3 + (1 / (n * K * (N - K) * (N - n) * (N - 2) * (N - 3))) * ((N - 1) * N * N * (N * (N + 1) - 6 * K * (N - K) - 6 * n * (N - n)) + 6 * n * K * (N - K) * (N - n) * (5 * N - 6));
                },
                median: function (N, K, n) {
                    return dists.hypergeometric.measurements.ppf(0.5, N, K, n);
                },
                mode: function (N, K, n) {
                    return Math.floor((n + 1) * (K + 1) / (N + 2));
                },
            },
        }
    }
}
console.log(dists.hypergeometric.measurements.stats.mean(70, 50, 40))
console.log(dists.hypergeometric.measurements.stats.variance(70, 50, 40))
console.log(dists.hypergeometric.measurements.stats.standardDeviation(70, 50, 40))
console.log(dists.hypergeometric.measurements.stats.skewness(70, 50, 40))
console.log(dists.hypergeometric.measurements.stats.kurtosis(70, 50, 40))
// console.log(dists.hypergeometric.measurements.stats.median(70, 50, 40))
console.log(dists.hypergeometric.measurements.stats.mode(70, 50, 40))

// mean_value: 28.571428571428573
// variance_value: 3.54924578527063
// standard_deviation_value: 1.8839442097022485
// skewness_value: 0.033453862253168605
// kurtosis_value: 2.9440732001755925
// median_value: #REF!
// mode_value: 29