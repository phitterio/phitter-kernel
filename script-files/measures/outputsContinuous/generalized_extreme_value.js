jStat = require("../node_modules/jstat");

dists = {
    generalized_extreme_value: {
        measurements: {
            nonCentralMoments: function (k, xi, mu, sigma) {
                return jStat.gammafn(1 - xi * k)
            },
            centralMoments: function (k, xi, mu, sigma) {
                const miu1 = this.nonCentralMoments(1, xi, mu, sigma);
                const miu2 = this.nonCentralMoments(2, xi, mu, sigma);
                const miu3 = this.nonCentralMoments(3, xi, mu, sigma);
                const miu4 = this.nonCentralMoments(4, xi, mu, sigma);

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
                mean: function (xi, mu, sigma) {
                    const miu1 = dists.generalized_extreme_value.measurements.nonCentralMoments(1, xi, mu, sigma);
                    if (xi == 0) { return mu + sigma * 0.5772156649 };
                    return mu + sigma * (miu1 - 1) / xi;
                },
                variance: function (xi, mu, sigma) {
                    const miu1 = dists.generalized_extreme_value.measurements.nonCentralMoments(1, xi, mu, sigma);
                    const miu2 = dists.generalized_extreme_value.measurements.nonCentralMoments(2, xi, mu, sigma);
                    if (xi == 0) { return (sigma ** 2) * ((Math.PI ** 2) / 6) };
                    return (sigma ** 2) * (miu2 - miu1 ** 2) / (xi ** 2);
                },
                standardDeviation: function (xi, mu, sigma) {
                    return Math.sqrt(this.variance(xi, mu, sigma));
                },
                skewness: function (xi, mu, sigma) {
                    const central_miu3 = dists.generalized_extreme_value.measurements.centralMoments(3, xi, mu, sigma);
                    const miu1 = dists.generalized_extreme_value.measurements.nonCentralMoments(1, xi, mu, sigma);
                    const miu2 = dists.generalized_extreme_value.measurements.nonCentralMoments(2, xi, mu, sigma);
                    const std = Math.sqrt(miu2 - miu1 ** 2);
                    if (xi == 0) { return 12 * Math.sqrt(6) * 1.20205690315959 / (Math.PI ** 3) };
                    return central_miu3 / (std ** 3);
                },
                kurtosis: function (xi, mu, sigma) {
                    const central_miu4 = dists.generalized_extreme_value.measurements.centralMoments(4, xi, mu, sigma);
                    const miu1 = dists.generalized_extreme_value.measurements.nonCentralMoments(1, xi, mu, sigma);
                    const miu2 = dists.generalized_extreme_value.measurements.nonCentralMoments(2, xi, mu, sigma);
                    const std = Math.sqrt(miu2 - miu1 ** 2);
                    if (xi == 0) { return 5.4 };
                    return central_miu4 / (std ** 4);
                },
                median: function (xi, mu, sigma) {
                    return dists.generalized_extreme_value.measurements.ppf(0.5, xi, mu, sigma);
                },
                mode: function (xi, mu, sigma) {
                    if (xi == 0) { return mu };
                    return mu + sigma * ((1 + xi) ** (-xi) - 1) / xi;
                },
            },
        }
    }
}
console.log(dists.generalized_extreme_value.measurements.stats.mean(0.1, 100, 5))
console.log(dists.generalized_extreme_value.measurements.stats.variance(0.1, 100, 5))
console.log(dists.generalized_extreme_value.measurements.stats.standardDeviation(0.1, 100, 5))
console.log(dists.generalized_extreme_value.measurements.stats.skewness(0.1, 100, 5))
console.log(dists.generalized_extreme_value.measurements.stats.kurtosis(0.1, 100, 5))
// console.log(dists.generalized_extreme_value.measurements.stats.median(0.1, 100, 5))
console.log(dists.generalized_extreme_value.measurements.stats.mode(0.1, 100, 5))

// mean_value: 103.43143510596599
// variance_value: 55.65602683020475
// standard_deviation_value: 7.460296698537181
// skewness_value: 1.910339134168694
// kurtosis_value: 10.978566239356496
// median_value: 101.86656160617854
// mode_value: 99.52571291072609