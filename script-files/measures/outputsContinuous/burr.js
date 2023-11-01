jStat = require("../node_modules/jstat");

dists = {
    burr: {
        measurements: {
            nonCentralMoments: function (k, A, B, C) {
                return (A ** k) * (C) * (jStat.gammafn((B * C - k) / B) * jStat.gammafn((B + k) / B) / jStat.gammafn(((B * C - k) / B) + ((B + k) / B)))
            },
            centralMoments: function (k, A, B, C) {
                const miu1 = this.nonCentralMoments(1, A, B, C);
                const miu2 = this.nonCentralMoments(2, A, B, C);
                const miu3 = this.nonCentralMoments(3, A, B, C);
                const miu4 = this.nonCentralMoments(4, A, B, C);

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
                mean: function (A, B, C) {
                    const miu1 = dists.burr.measurements.nonCentralMoments(1, A, B, C);
                    return miu1;
                },
                variance: function (A, B, C) {
                    const miu1 = dists.burr.measurements.nonCentralMoments(1, A, B, C);
                    const miu2 = dists.burr.measurements.nonCentralMoments(2, A, B, C);
                    return miu2 - miu1 ** 2;
                },
                standardDeviation: function (A, B, C) {
                    return Math.sqrt(this.variance(A, B, C));
                },
                skewness: function (A, B, C) {
                    const central_miu3 = dists.burr.measurements.centralMoments(3, A, B, C);
                    return central_miu3 / (this.standardDeviation(A, B, C) ** 3);
                },
                kurtosis: function (A, B, C) {
                    const central_miu4 = dists.burr.measurements.centralMoments(4, A, B, C);
                    return central_miu4 / (this.standardDeviation(A, B, C) ** 4);
                },
                median: function (A, B, C) {
                    return dists.burr.measurements.ppf(0.5, A, B, C);
                },
                mode: function (A, B, C) {
                    return A * ((B - 1) / (B * C + 1)) ** (1 / B);
                },
            },
        }
    }
}
console.log(dists.burr.measurements.stats.mean(2, 100, 0.5))
console.log(dists.burr.measurements.stats.variance(2, 100, 0.5))
console.log(dists.burr.measurements.stats.standardDeviation(2, 100, 0.5))
console.log(dists.burr.measurements.stats.skewness(2, 100, 0.5))
console.log(dists.burr.measurements.stats.kurtosis(2, 100, 0.5))
// console.log(dists.burr.measurements.stats.median(2, 100, 0.5))
console.log(dists.burr.measurements.stats.mode(2, 100, 0.5))

// mean_value: 2.0285911939977317
// variance_value: 0.002770547685345015
// standard_deviation_value: 0.05263599229942393
// skewness_value: 1.000379300692667
// kurtosis_value: 5.975370190030128
// median_value: 2.022093383875707
// mode_value: 2.013309977705743