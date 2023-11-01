jStat = require("../node_modules/jstat");

dists = {
    pareto_second_kind: {
        measurements: {
            nonCentralMoments: function (k, xm, alpha, loc) {
                return (xm ** k * jStat.gammafn(alpha - k) * jStat.gammafn(1 + k)) / (jStat.gammafn(alpha))
            },
            centralMoments: function (k, xm, alpha, loc) {
                const miu1 = this.nonCentralMoments(1, xm, alpha, loc);
                const miu2 = this.nonCentralMoments(2, xm, alpha, loc);
                const miu3 = this.nonCentralMoments(3, xm, alpha, loc);
                const miu4 = this.nonCentralMoments(4, xm, alpha, loc);

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
                mean: function (xm, alpha, loc) {
                    const miu1 = dists.pareto_second_kind.measurements.nonCentralMoments(1, xm, alpha, loc);
                    return loc + miu1;
                },
                variance: function (xm, alpha, loc) {
                    const miu1 = dists.pareto_second_kind.measurements.nonCentralMoments(1, xm, alpha, loc);
                    const miu2 = dists.pareto_second_kind.measurements.nonCentralMoments(2, xm, alpha, loc);
                    return miu2 - miu1 ** 2;
                },
                standardDeviation: function (xm, alpha, loc) {
                    return Math.sqrt(this.variance(xm, alpha, loc));
                },
                skewness: function (xm, alpha, loc) {
                    const central_miu3 = dists.pareto_second_kind.measurements.centralMoments(3, xm, alpha, loc);
                    return central_miu3 / (this.standardDeviation(xm, alpha, loc) ** 3);
                },
                kurtosis: function (xm, alpha, loc) {
                    const central_miu4 = dists.pareto_second_kind.measurements.centralMoments(4, xm, alpha, loc);
                    return central_miu4 / (this.standardDeviation(xm, alpha, loc) ** 4);
                },
                median: function (xm, alpha, loc) {
                    return dists.pareto_second_kind.measurements.ppf(0.5, xm, alpha, loc);
                },
                mode: function (xm, alpha, loc) {
                    return loc;
                },
            },
        }
    }
}
console.log(dists.pareto_second_kind.measurements.stats.mean(32, 7, 17))
console.log(dists.pareto_second_kind.measurements.stats.variance(32, 7, 17))
console.log(dists.pareto_second_kind.measurements.stats.standardDeviation(32, 7, 17))
console.log(dists.pareto_second_kind.measurements.stats.skewness(32, 7, 17))
console.log(dists.pareto_second_kind.measurements.stats.kurtosis(32, 7, 17))
// console.log(dists.pareto_second_kind.measurements.stats.median(32, 7, 17))
console.log(dists.pareto_second_kind.measurements.stats.mode(32, 7, 17))

// mean_value: 22.333333333333332
// variance_value: 39.82222222222222
// standard_deviation_value: 6.310485101972924
// skewness_value: 3.380617018914066
// kurtosis_value: 27.857142857142843
// median_value: 20.330864437561992
// mode_value: 17