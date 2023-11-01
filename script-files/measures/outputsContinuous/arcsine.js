jStat = require("../node_modules/jstat");

dists = {
    arcsine: {
        measurements: {
            nonCentralMoments: function (k, a, b) {
                return jStat.gammafn(0.5) * jStat.gammafn(k + 0.5) / (Math.PI * jStat.gammafn(k + 1))
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
                    const miu1 = dists.arcsine.measurements.nonCentralMoments(1, a, b);
                    return miu1 * (b - a) + a;
                },
                variance: function (a, b) {
                    const miu1 = dists.arcsine.measurements.nonCentralMoments(1, a, b);
                    const miu2 = dists.arcsine.measurements.nonCentralMoments(2, a, b);
                    return (miu2 - miu1 ** 2) * (b - a) ** 2;
                },
                standardDeviation: function (a, b) {
                    return Math.sqrt(this.variance(a, b));
                },
                skewness: function (a, b) {
                    const central_miu3 = dists.arcsine.measurements.centralMoments(3, a, b);
                    const miu1 = dists.arcsine.measurements.nonCentralMoments(1, a, b);
                    const miu2 = dists.arcsine.measurements.nonCentralMoments(2, a, b);
                    const std = Math.sqrt(miu2 - miu1 ** 2);
                    return central_miu3 / (std ** 3);
                },
                kurtosis: function (a, b) {
                    const central_miu4 = dists.arcsine.measurements.centralMoments(4, a, b);
                    const miu1 = dists.arcsine.measurements.nonCentralMoments(1, a, b);
                    const miu2 = dists.arcsine.measurements.nonCentralMoments(2, a, b);
                    const std = Math.sqrt(miu2 - miu1 ** 2);
                    return central_miu4 / (std ** 4);
                },
                median: function (a, b) {
                    return dists.arcsine.measurements.ppf(0.5, a, b);
                },
                mode: function (a, b) {
                    return undefined;
                },
            },
        }
    }
}
console.log(dists.arcsine.measurements.stats.mean(78, 89))
console.log(dists.arcsine.measurements.stats.variance(78, 89))
console.log(dists.arcsine.measurements.stats.standardDeviation(78, 89))
console.log(dists.arcsine.measurements.stats.skewness(78, 89))
console.log(dists.arcsine.measurements.stats.kurtosis(78, 89))
// console.log(dists.arcsine.measurements.stats.median(78, 89))
console.log(dists.arcsine.measurements.stats.mode(78, 89))

// mean_value: 83.5
// variance_value: 15.124999999999986
// standard_deviation_value: 3.8890872965260095
// skewness_value: 0
// kurtosis_value: 1.5000000000000027
// median_value: 83.5
// mode_value: undefined