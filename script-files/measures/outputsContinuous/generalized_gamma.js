jStat = require("../node_modules/jstat");

dists = {
    generalized_gamma: {
        measurements: {
            nonCentralMoments: function (k, a, d, p) {
                return (a ** k) * jStat.gammafn((d + k) / p) / jStat.gammafn((d) / p)
            },
            centralMoments: function (k, a, d, p) {
                const µ1 = this.nonCentralMoments(1, a, d, p);
                const µ2 = this.nonCentralMoments(2, a, d, p);
                const µ3 = this.nonCentralMoments(3, a, d, p);
                const µ4 = this.nonCentralMoments(4, a, d, p);

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
                mean: function (a, d, p) {
                    const µ1 = dists.generalized_gamma.measurements.nonCentralMoments(1, a, d, p);
                    return µ1;
                },
                variance: function (a, d, p) {
                    const µ1 = dists.generalized_gamma.measurements.nonCentralMoments(1, a, d, p);
                    const µ2 = dists.generalized_gamma.measurements.nonCentralMoments(2, a, d, p);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (a, d, p) {
                    return Math.sqrt(this.variance(a, d, p));
                },
                skewness: function (a, d, p) {
                    const central_µ3 = dists.generalized_gamma.measurements.centralMoments(3, a, d, p);
                    return central_µ3 / (this.standardDeviation(a, d, p) ** 3);
                },
                kurtosis: function (a, d, p) {
                    const central_µ4 = dists.generalized_gamma.measurements.centralMoments(4, a, d, p);
                    return central_µ4 / (this.standardDeviation(a, d, p) ** 4);
                },
                median: function (a, d, p) {
                    return dists.generalized_gamma.measurements.ppf(0.5, a, d, p);
                },
                mode: function (a, d, p) {
                    return a * ((d - 1) / p) ** (1 / p);
                },
            },
        }
    }
}
console.log(dists.generalized_gamma.measurements.stats.mean(10, 150, 20))
console.log(dists.generalized_gamma.measurements.stats.variance(10, 150, 20))
console.log(dists.generalized_gamma.measurements.stats.standardDeviation(10, 150, 20))
console.log(dists.generalized_gamma.measurements.stats.skewness(10, 150, 20))
console.log(dists.generalized_gamma.measurements.stats.kurtosis(10, 150, 20))
// console.log(dists.generalized_gamma.measurements.stats.median(10, 150, 20))
console.log(dists.generalized_gamma.measurements.stats.mode(10, 150, 20))

// mean_value: 11.024287744229486
// variance_value: 0.04303370606395163
// standard_deviation_value: 0.20744567014992535
// skewness_value: -0.4409436687911422
// kurtosis_value: 3.1980826667115974
// median_value: 11.03504810450817
// mode_value: 11.056249067821739