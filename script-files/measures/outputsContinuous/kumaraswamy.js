jStat = require("../node_modules/jstat");

dists = {
    kumaraswamy: {
        measurements: {
            nonCentralMoments: function (k, alpha, beta, min, max) {
                return beta * jStat.gammafn(1 + k / alpha) * jStat.gammafn(beta) / jStat.gammafn(1 + beta + k / alpha)
            },
            centralMoments: function (k, alpha, beta, min, max) {
                const miu1 = this.nonCentralMoments(1, alpha, beta, min, max);
                const miu2 = this.nonCentralMoments(2, alpha, beta, min, max);
                const miu3 = this.nonCentralMoments(3, alpha, beta, min, max);
                const miu4 = this.nonCentralMoments(4, alpha, beta, min, max);

                let result;
                switch (k) {
                    case 1: result = 0; break;
                    case 2: result = miu2 - miu1 ** 2; break;
                    case 3: result = miu3 - 3 * miu1 * miu2 + 2 * miu1 ** 3; break;
                    case 4: result = miu4 - 4 * miu1 * miu3 + 6 * (miu1 ** 2) * miu2 - 3 * (miu1 ** 4); break;
                };
                return result
            },
            stats: {
                mean: function (alpha, beta, min, max) {
                    const miu1 = dists.kumaraswamy.measurements.nonCentralMoments(1, alpha, beta, min, max);
                    return min + (max - min) * miu1;
                },
                variance: function (alpha, beta, min, max) {
                    const miu1 = dists.kumaraswamy.measurements.nonCentralMoments(1, alpha, beta, min, max);
                    const miu2 = dists.kumaraswamy.measurements.nonCentralMoments(2, alpha, beta, min, max);
                    return (max - min) ** 2 * (miu2 - miu1 ** 2);
                },
                standardDeviation: function (alpha, beta, min, max) {
                    return Math.sqrt(this.variance(alpha, beta, min, max));
                },
                skewness: function (alpha, beta, min, max) {
                    const central_miu3 = dists.kumaraswamy.measurements.centralMoments(3, alpha, beta, min, max);
                    const miu1 = dists.kumaraswamy.measurements.nonCentralMoments(1, alpha, beta, min, max);
                    const miu2 = dists.kumaraswamy.measurements.nonCentralMoments(2, alpha, beta, min, max);
                    const std = Math.sqrt(miu2 - miu1 ** 2);
                    return central_miu3 / (std ** 3);
                },
                kurtosis: function (alpha, beta, min, max) {
                    const central_miu4 = dists.kumaraswamy.measurements.centralMoments(4, alpha, beta, min, max);
                    const miu1 = dists.kumaraswamy.measurements.nonCentralMoments(1, alpha, beta, min, max);
                    const miu2 = dists.kumaraswamy.measurements.nonCentralMoments(2, alpha, beta, min, max);
                    const std = Math.sqrt(miu2 - miu1 ** 2);
                    return central_miu4 / (std ** 4);
                },
                median: function (alpha, beta, min, max) {
                    return dists.kumaraswamy.measurements.ppf(0.5, alpha, beta, min, max);
                },
                mode: function (alpha, beta, min, max) {
                    return ((alpha - 1) / (alpha * beta - 1)) ** (1 / alpha) * min + (max - min);
                },
            },
        }
    }
}
console.log(dists.kumaraswamy.measurements.stats.mean(10, 10, 10, 20))
console.log(dists.kumaraswamy.measurements.stats.variance(10, 10, 10, 20))
console.log(dists.kumaraswamy.measurements.stats.standardDeviation(10, 10, 10, 20))
console.log(dists.kumaraswamy.measurements.stats.skewness(10, 10, 10, 20))
console.log(dists.kumaraswamy.measurements.stats.kurtosis(10, 10, 10, 20))
// console.log(dists.kumaraswamy.measurements.stats.median(10, 10, 10, 20))
console.log(dists.kumaraswamy.measurements.stats.mode(10, 10, 10, 20))

// mean_value: 17.516217822684155
// variance_value: 0.7637157807332429
// standard_deviation_value: 0.8739083365738325
// skewness_value: -0.7489330799371784
// kurtosis_value: 3.71508174002282
// median_value: 17.6310814651777
// mode_value: 17.867934421967725