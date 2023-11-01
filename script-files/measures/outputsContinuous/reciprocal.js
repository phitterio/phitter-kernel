jStat = require("../node_modules/jstat");

dists = {
    reciprocal: {
        measurements: {
            nonCentralMoments: function (k, a, b) {
                return (b ** k - a ** k) / (k * (Math.log(b) - Math.log(a)))
            },
            centralMoments: function (k, a, b) {
                const miu1 = this.nonCentralMoments(1, a, b);
                const miu2 = this.nonCentralMoments(2, a, b);
                const miu3 = this.nonCentralMoments(3, a, b);
                const miu4 = this.nonCentralMoments(4, a, b);

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
                mean: function (a, b) {
                    const miu1 = dists.reciprocal.measurements.nonCentralMoments(1, a, b);
                    return miu1;
                },
                variance: function (a, b) {
                    const miu1 = dists.reciprocal.measurements.nonCentralMoments(1, a, b);
                    const miu2 = dists.reciprocal.measurements.nonCentralMoments(2, a, b);
                    return miu2 - miu1 ** 2;
                },
                standardDeviation: function (a, b) {
                    return Math.sqrt(this.variance(a, b));
                },
                skewness: function (a, b) {
                    const central_miu3 = dists.reciprocal.measurements.centralMoments(3, a, b);
                    return central_miu3 / (this.standardDeviation(a, b) ** 3);
                },
                kurtosis: function (a, b) {
                    const central_miu4 = dists.reciprocal.measurements.centralMoments(4, a, b);
                    return central_miu4 / (this.standardDeviation(a, b) ** 4);
                },
                median: function (a, b) {
                    return dists.reciprocal.measurements.ppf(0.5, a, b);
                },
                mode: function (a, b) {
                    return a;
                },
            },
        }
    }
}
console.log(dists.reciprocal.measurements.stats.mean(20, 100))
console.log(dists.reciprocal.measurements.stats.variance(20, 100))
console.log(dists.reciprocal.measurements.stats.standardDeviation(20, 100))
console.log(dists.reciprocal.measurements.stats.skewness(20, 100))
console.log(dists.reciprocal.measurements.stats.kurtosis(20, 100))
// console.log(dists.reciprocal.measurements.stats.median(20, 100))
console.log(dists.reciprocal.measurements.stats.mode(20, 100))

// mean_value: 49.70679476476893
// variance_value: 511.642240099276
// standard_deviation_value: 22.61951016488368
// skewness_value: 0.5482226050475848
// kurtosis_value: 2.1295341717725487
// median_value: 44.72135954999579
// mode_value: 20