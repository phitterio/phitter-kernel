jStat = require("../node_modules/jstat");

dists = {
    erlang: {
        measurements: {
            nonCentralMoments: function (k, k_, beta) {
                return beta ** k * (jStat.gammafn(k + k_) / (jStat.factorial(k_ - 1)))
            },
            centralMoments: function (k, k, beta) {
                const µ1 = this.nonCentralMoments(1, k, beta);
                const µ2 = this.nonCentralMoments(2, k, beta);
                const µ3 = this.nonCentralMoments(3, k, beta);
                const µ4 = this.nonCentralMoments(4, k, beta);

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
                mean: function (k, beta) {
                    const µ1 = dists.erlang.measurements.nonCentralMoments(1, k, beta);
                    return µ1;
                },
                variance: function (k, beta) {
                    const µ1 = dists.erlang.measurements.nonCentralMoments(1, k, beta);
                    const µ2 = dists.erlang.measurements.nonCentralMoments(2, k, beta);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (k, beta) {
                    return Math.sqrt(this.variance(k, beta));
                },
                skewness: function (k, beta) {
                    const central_µ3 = dists.erlang.measurements.centralMoments(3, k, beta);
                    return central_µ3 / (this.standardDeviation(k, beta) ** 3);
                },
                kurtosis: function (k, beta) {
                    const central_µ4 = dists.erlang.measurements.centralMoments(4, k, beta);
                    return central_µ4 / (this.standardDeviation(k, beta) ** 4);
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