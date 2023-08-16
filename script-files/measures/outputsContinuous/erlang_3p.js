jStat = require("../node_modules/jstat");

dists = {
    erlang_3p: {
        measurements: {
            nonCentralMoments: function (k, k_, beta, loc) {
                return beta ** k * (jStat.gammafn(k_ + k) / (jStat.factorial(k_ - 1)))
            },
            centralMoments: function (k, k_, beta, loc) {
                const µ1 = this.nonCentralMoments(1, k_, beta, loc);
                const µ2 = this.nonCentralMoments(2, k_, beta, loc);
                const µ3 = this.nonCentralMoments(3, k_, beta, loc);
                const µ4 = this.nonCentralMoments(4, k_, beta, loc);

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
                mean: function (k, beta, loc) {
                    const µ1 = dists.erlang_3p.measurements.nonCentralMoments(1, k, beta, loc);
                    return loc + µ1;
                },
                variance: function (k, beta, loc) {
                    const µ1 = dists.erlang_3p.measurements.nonCentralMoments(1, k, beta, loc);
                    const µ2 = dists.erlang_3p.measurements.nonCentralMoments(2, k, beta, loc);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (k, beta, loc) {
                    return Math.sqrt(this.variance(k, beta, loc));
                },
                skewness: function (k, beta, loc) {
                    const central_µ3 = dists.erlang_3p.measurements.centralMoments(3, k, beta, loc);
                    return central_µ3 / (this.standardDeviation(k, beta, loc) ** 3);
                },
                kurtosis: function (k, beta, loc) {
                    const central_µ4 = dists.erlang_3p.measurements.centralMoments(4, k, beta, loc);
                    return central_µ4 / (this.standardDeviation(k, beta, loc) ** 4);
                },
                median: function (k, beta, loc) {
                    return dists.erlang_3p.measurements.ppf(0.5, k, beta, loc);
                },
                mode: function (k, beta, loc) {
                    return beta * (k - 1) + loc;
                },
            },
        }
    }
}
console.log(dists.erlang_3p.measurements.stats.mean(50, 6, 1000))
console.log(dists.erlang_3p.measurements.stats.variance(50, 6, 1000))
console.log(dists.erlang_3p.measurements.stats.standardDeviation(50, 6, 1000))
console.log(dists.erlang_3p.measurements.stats.skewness(50, 6, 1000))
console.log(dists.erlang_3p.measurements.stats.kurtosis(50, 6, 1000))
// console.log(dists.erlang_3p.measurements.stats.median(50, 6, 1000))
console.log(dists.erlang_3p.measurements.stats.mode(50, 6, 1000))

// mean_value: 1299.9999999999998
// variance_value: 1800.0000000000146
// standard_deviation_value: 42.426406871193024
// skewness_value: 0.2828427124749082
// kurtosis_value: 3.1199999999928854
// median_value: 1298.0023877079652
// mode_value: 1294