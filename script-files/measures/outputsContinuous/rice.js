jStat = require("../node_modules/jstat");
BESSEL = require("../node_modules/bessel");

dists = {
    rice: {
        measurements: {
            nonCentralMoments: function (k, v, sigma) {
                let result;
                switch (k) {
                    case 1: result = sigma * Math.sqrt(Math.PI / 2) * Math.exp((-v * v / (2 * sigma * sigma)) / 2) * ((1 - (-v * v / (2 * sigma * sigma))) * BESSEL.besseli((-v * v / (4 * sigma * sigma)), 0) + (-v * v / (2 * sigma * sigma)) * BESSEL.besseli((-v * v / (4 * sigma * sigma)), 1)); break;
                    case 2: result = 2 * sigma * sigma + v * v; break;
                    case 3: result = 3 * sigma ** 3 * Math.sqrt(Math.PI / 2) * Math.exp((-v * v / (2 * sigma * sigma)) / 2) * ((2 * (-v * v / (2 * sigma * sigma)) ** 2 - 6 * (-v * v / (2 * sigma * sigma)) + 3) * BESSEL.besseli(-v * v / (4 * sigma * sigma), 0) - 2 * ((-v * v / (2 * sigma * sigma)) - 2) * (-v * v / (2 * sigma * sigma)) * BESSEL.besseli(-v * v / (4 * sigma * sigma), 1)) / 3; break;
                    case 4: result = 8 * sigma ** 4 + 8 * sigma * sigma * v * v + v ** 4; break;
                };
                return result
            },
            centralMoments: function (k, v, sigma) {
                const µ1 = this.nonCentralMoments(1, v, sigma);
                const µ2 = this.nonCentralMoments(2, v, sigma);
                const µ3 = this.nonCentralMoments(3, v, sigma);
                const µ4 = this.nonCentralMoments(4, v, sigma);

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
                mean: function (v, sigma) {
                    const µ1 = dists.rice.measurements.nonCentralMoments(1, v, sigma);
                    return µ1;
                },
                variance: function (v, sigma) {
                    const µ1 = dists.rice.measurements.nonCentralMoments(1, v, sigma);
                    const µ2 = dists.rice.measurements.nonCentralMoments(2, v, sigma);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (v, sigma) {
                    return Math.sqrt(this.variance(v, sigma));
                },
                skewness: function (v, sigma) {
                    const central_µ3 = dists.rice.measurements.centralMoments(3, v, sigma);
                    return central_µ3 / (this.standardDeviation(v, sigma) ** 3);
                },
                kurtosis: function (v, sigma) {
                    const central_µ4 = dists.rice.measurements.centralMoments(4, v, sigma);
                    return central_µ4 / (this.standardDeviation(v, sigma) ** 4);
                },
                median: function (v, sigma) {
                    return dists.rice.measurements.ppf(0.5, v, sigma);
                },
                mode: function (v, sigma) {
                    return undefined;
                },
            },
        }
    }
}
console.log(dists.rice.measurements.stats.mean(5, 5))
console.log(dists.rice.measurements.stats.variance(5, 5))
console.log(dists.rice.measurements.stats.standardDeviation(5, 5))
console.log(dists.rice.measurements.stats.skewness(5, 5))
console.log(dists.rice.measurements.stats.kurtosis(5, 5))
// console.log(dists.rice.measurements.stats.median(5, 5))
console.log(dists.rice.measurements.stats.mode(5, 5))

// mean_value: 7.7428622389271755
// variance_value: 15.048084348995644
// standard_deviation_value: 3.8791860420706357
// skewness_value: 0.5171538009483962
// kurtosis_value: 3.0153802046033173
// median_value: -
// mode_value: -