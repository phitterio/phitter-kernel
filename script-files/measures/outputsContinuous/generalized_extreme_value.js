jStat = require("../node_modules/jstat");

dists = {
    generalized_extreme_value: {
        measurements: {
            nonCentralMoments: function (k, ξ, miu, sigma) {
                return jStat.gammafn(1 - ξ * k)
            },
            centralMoments: function (k, ξ, miu, sigma) {
                const µ1 = this.nonCentralMoments(1, ξ, miu, sigma);
                const µ2 = this.nonCentralMoments(2, ξ, miu, sigma);
                const µ3 = this.nonCentralMoments(3, ξ, miu, sigma);
                const µ4 = this.nonCentralMoments(4, ξ, miu, sigma);

                let result;
                switch (k) {
                    case 1: result = 0; break;
                    case 2: result = µ2 - µ1 ** 2; break;
                    case 3: result = µ3 - 3 * µ1 * µ2 + 2 * µ1 ** 3; break;
                    case 4: result = µ4 - 4 * µ1 * µ3 + 6 * (µ1 ** 2) * µ2 - 3 * (µ1 ** 4); break;
                };
                return result
            },
            stats: {
                mean: function (ξ, miu, sigma) {
                    const µ1 = dists.generalized_extreme_value.measurements.nonCentralMoments(1, ξ, miu, sigma);
                    if (ξ == 0) { return miu + sigma * 0.5772156649 };
                    return miu + sigma * (µ1 - 1) / ξ;
                },
                variance: function (ξ, miu, sigma) {
                    const µ1 = dists.generalized_extreme_value.measurements.nonCentralMoments(1, ξ, miu, sigma);
                    const µ2 = dists.generalized_extreme_value.measurements.nonCentralMoments(2, ξ, miu, sigma);
                    if (ξ == 0) { return (sigma ** 2) * ((Math.PI ** 2) / 6) };
                    return (sigma ** 2) * (µ2 - µ1 ** 2) / (ξ ** 2);
                },
                standardDeviation: function (ξ, miu, sigma) {
                    return Math.sqrt(this.variance(ξ, miu, sigma));
                },
                skewness: function (ξ, miu, sigma) {
                    const central_µ3 = dists.generalized_extreme_value.measurements.centralMoments(3, ξ, miu, sigma);
                    const µ1 = dists.generalized_extreme_value.measurements.nonCentralMoments(1, ξ, miu, sigma);
                    const µ2 = dists.generalized_extreme_value.measurements.nonCentralMoments(2, ξ, miu, sigma);
                    const std = Math.sqrt(µ2 - µ1 ** 2);
                    if (ξ == 0) { return 12 * Math.sqrt(6) * 1.20205690315959 / (Math.PI ** 3) };
                    return central_µ3 / (std ** 3);
                },
                kurtosis: function (ξ, miu, sigma) {
                    const central_µ4 = dists.generalized_extreme_value.measurements.centralMoments(4, ξ, miu, sigma);
                    const µ1 = dists.generalized_extreme_value.measurements.nonCentralMoments(1, ξ, miu, sigma);
                    const µ2 = dists.generalized_extreme_value.measurements.nonCentralMoments(2, ξ, miu, sigma);
                    const std = Math.sqrt(µ2 - µ1 ** 2);
                    if (ξ == 0) { return 5.4 };
                    return central_µ4 / (std ** 4);
                },
                median: function (ξ, miu, sigma) {
                    return dists.generalized_extreme_value.measurements.ppf(0.5, ξ, miu, sigma);
                },
                mode: function (ξ, miu, sigma) {
                    if (ξ == 0) { return miu };
                    return miu + sigma * ((1 + ξ) ** (-ξ) - 1) / ξ;
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