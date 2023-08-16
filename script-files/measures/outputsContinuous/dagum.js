jStat = require("../node_modules/jstat");

dists = {
    dagum: {
        measurements: {
            nonCentralMoments: function (k, a, b, p) {
                return (b ** k) * (p) * (jStat.gammafn((a * p + k) / a) * jStat.gammafn((a - k) / a) / jStat.gammafn(((a * p + k) / a) + ((a - k) / a)))
            },
            centralMoments: function (k, a, b, p) {
                const µ1 = this.nonCentralMoments(1, a, b, p);
                const µ2 = this.nonCentralMoments(2, a, b, p);
                const µ3 = this.nonCentralMoments(3, a, b, p);
                const µ4 = this.nonCentralMoments(4, a, b, p);

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
                mean: function (a, b, p) {
                    const µ1 = dists.dagum.measurements.nonCentralMoments(1, a, b, p);
                    return µ1;
                },
                variance: function (a, b, p) {
                    const µ1 = dists.dagum.measurements.nonCentralMoments(1, a, b, p);
                    const µ2 = dists.dagum.measurements.nonCentralMoments(2, a, b, p);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (a, b, p) {
                    return Math.sqrt(this.variance(a, b, p));
                },
                skewness: function (a, b, p) {
                    const central_µ3 = dists.dagum.measurements.centralMoments(3, a, b, p);
                    return central_µ3 / (this.standardDeviation(a, b, p) ** 3);
                },
                kurtosis: function (a, b, p) {
                    const central_µ4 = dists.dagum.measurements.centralMoments(4, a, b, p);
                    return central_µ4 / (this.standardDeviation(a, b, p) ** 4);
                },
                median: function (a, b, p) {
                    return dists.dagum.measurements.ppf(0.5, a, b, p);
                },
                mode: function (a, b, p) {
                    return b * ((a * p - 1) / (a + 1)) ** (1 / a);
                },
            },
        }
    }
}
console.log(dists.dagum.measurements.stats.mean(8, 50, 3))
console.log(dists.dagum.measurements.stats.variance(8, 50, 3))
console.log(dists.dagum.measurements.stats.standardDeviation(8, 50, 3))
console.log(dists.dagum.measurements.stats.skewness(8, 50, 3))
console.log(dists.dagum.measurements.stats.kurtosis(8, 50, 3))
// console.log(dists.dagum.measurements.stats.median(8, 50, 3))
console.log(dists.dagum.measurements.stats.mode(8, 50, 3))

// mean_value: 61.329820080267886
// variance_value: 143.53075128772343
// standard_deviation_value: 11.980432015905079
// skewness_value: 1.8446259989622031
// kurtosis_value: 11.78027670241988
// median_value: 59.17180573505291
// mode_value: 56.22191965196019