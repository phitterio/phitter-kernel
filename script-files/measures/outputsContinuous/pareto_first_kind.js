jStat = require("../node_modules/jstat");

dists = {
    pareto_first_kind: {
        measurements: {
            nonCentralMoments: function (k, xm, alpha, loc) {
                return (alpha * xm ** k) / (alpha - k)
            },
            centralMoments: function (k, xm, alpha, loc) {
                const µ1 = this.nonCentralMoments(1, xm, alpha, loc);
                const µ2 = this.nonCentralMoments(2, xm, alpha, loc);
                const µ3 = this.nonCentralMoments(3, xm, alpha, loc);
                const µ4 = this.nonCentralMoments(4, xm, alpha, loc);

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
                mean: function (xm, alpha, loc) {
                    const µ1 = dists.pareto_first_kind.measurements.nonCentralMoments(1, xm, alpha, loc);
                    return loc + µ1;
                },
                variance: function (xm, alpha, loc) {
                    const µ1 = dists.pareto_first_kind.measurements.nonCentralMoments(1, xm, alpha, loc);
                    const µ2 = dists.pareto_first_kind.measurements.nonCentralMoments(2, xm, alpha, loc);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (xm, alpha, loc) {
                    return Math.sqrt(this.variance(xm, alpha, loc));
                },
                skewness: function (xm, alpha, loc) {
                    const central_µ3 = dists.pareto_first_kind.measurements.centralMoments(3, xm, alpha, loc);
                    return central_µ3 / (this.standardDeviation(xm, alpha, loc) ** 3);
                },
                kurtosis: function (xm, alpha, loc) {
                    const central_µ4 = dists.pareto_first_kind.measurements.centralMoments(4, xm, alpha, loc);
                    return central_µ4 / (this.standardDeviation(xm, alpha, loc) ** 4);
                },
                median: function (xm, alpha, loc) {
                    return dists.pareto_first_kind.measurements.ppf(0.5, xm, alpha, loc);
                },
                mode: function (xm, alpha, loc) {
                    return xm + loc;
                },
            },
        }
    }
}
console.log(dists.pareto_first_kind.measurements.stats.mean(10, 7, 100))
console.log(dists.pareto_first_kind.measurements.stats.variance(10, 7, 100))
console.log(dists.pareto_first_kind.measurements.stats.standardDeviation(10, 7, 100))
console.log(dists.pareto_first_kind.measurements.stats.skewness(10, 7, 100))
console.log(dists.pareto_first_kind.measurements.stats.kurtosis(10, 7, 100))
// console.log(dists.pareto_first_kind.measurements.stats.median(10, 7, 100))
console.log(dists.pareto_first_kind.measurements.stats.mode(10, 7, 100))

// mean_value: 111.66666666666667
// variance_value: 3.888888888888914
// standard_deviation_value: 1.9720265943665451
// skewness_value: 3.380617018913936
// kurtosis_value: 27.857142857142687
// median_value: 111.04089513673813
// mode_value: 110