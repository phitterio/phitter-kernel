jStat = require("../node_modules/jstat");

dists = {
    erlang: {
        measurements: {
            nonCentralMoments: function (k, k_, beta) {
                return beta ** k * (jStat.gammafn(k + k_) / (jStat.factorial(k_ - 1)))
            },
            centralMoments: function (k, k, beta) {
                const miu1 = this.nonCentralMoments(1, k, beta);
                const miu2 = this.nonCentralMoments(2, k, beta);
                const miu3 = this.nonCentralMoments(3, k, beta);
                const miu4 = this.nonCentralMoments(4, k, beta);

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
                mean: function (k, beta) {
                    const miu1 = dists.erlang.measurements.nonCentralMoments(1, k, beta);
                    return miu1;
                },
                variance: function (k, beta) {
                    const miu1 = dists.erlang.measurements.nonCentralMoments(1, k, beta);
                    const miu2 = dists.erlang.measurements.nonCentralMoments(2, k, beta);
                    return miu2 - miu1 ** 2;
                },
                standardDeviation: function (k, beta) {
                    return Math.sqrt(this.variance(k, beta));
                },
                skewness: function (k, beta) {
                    const central_miu3 = dists.erlang.measurements.centralMoments(3, k, beta);
                    return central_miu3 / (this.standardDeviation(k, beta) ** 3);
                },
                kurtosis: function (k, beta) {
                    const central_miu4 = dists.erlang.measurements.centralMoments(4, k, beta);
                    return central_miu4 / (this.standardDeviation(k, beta) ** 4);
                },
                median: function (k, beta) {
                    return dists.erlang.measurements.ppf(0.5, k, beta);
                },
                mode: function (k, beta) {
                    return beta * (k - 1);
                },
            },
        }
    }
}
console.log(dists.erlang.measurements.stats.mean(78, 6))
console.log(dists.erlang.measurements.stats.variance(78, 6))
console.log(dists.erlang.measurements.stats.standardDeviation(78, 6))
console.log(dists.erlang.measurements.stats.skewness(78, 6))
console.log(dists.erlang.measurements.stats.kurtosis(78, 6))
// console.log(dists.erlang.measurements.stats.median(78, 6))
console.log(dists.erlang.measurements.stats.mode(78, 6))

// mean_value: 467.9999999999998
// variance_value: 2808.000000000262
// standard_deviation_value: 52.99056519796955
// skewness_value: 0.2264554068268846
// kurtosis_value: 3.0769230769573364
// median_value: 466.00152658811396
// mode_value: 462