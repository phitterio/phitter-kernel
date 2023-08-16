jStat = require("../node_modules/jstat");

dists = {
    beta_prime_4p: {
        measurements: {
            nonCentralMoments: function (k, alpha, beta, loc, scale) {
                return jStat.gammafn(k + alpha) * jStat.gammafn(beta - k) / (jStat.gammafn(alpha) * jStat.gammafn(beta))
            },
            centralMoments: function (k, alpha, beta, loc, scale) {
                const µ1 = this.nonCentralMoments(1, alpha, beta, loc, scale);
                const µ2 = this.nonCentralMoments(2, alpha, beta, loc, scale);
                const µ3 = this.nonCentralMoments(3, alpha, beta, loc, scale);
                const µ4 = this.nonCentralMoments(4, alpha, beta, loc, scale);

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
                mean: function (alpha, beta, loc, scale) {
                    const µ1 = dists.beta_prime_4p.measurements.nonCentralMoments(1, alpha, beta, loc, scale);
                    return loc + scale * µ1;
                },
                variance: function (alpha, beta, loc, scale) {
                    const µ1 = dists.beta_prime_4p.measurements.nonCentralMoments(1, alpha, beta, loc, scale);
                    const µ2 = dists.beta_prime_4p.measurements.nonCentralMoments(2, alpha, beta, loc, scale);
                    return (scale ** 2) * (µ2 - µ1 ** 2);
                },
                standardDeviation: function (alpha, beta, loc, scale) {
                    return Math.sqrt(this.variance(alpha, beta, loc, scale));
                },
                skewness: function (alpha, beta, loc, scale) {
                    const µ1 = dists.beta_prime_4p.measurements.nonCentralMoments(1, alpha, beta, loc, scale);
                    const µ2 = dists.beta_prime_4p.measurements.nonCentralMoments(2, alpha, beta, loc, scale);
                    const std = Math.sqrt(µ2 - µ1 ** 2);
                    const central_µ3 = dists.beta_prime_4p.measurements.centralMoments(3, alpha, beta, loc, scale);
                    return central_µ3 / (std ** 3);
                },
                kurtosis: function (alpha, beta, loc, scale) {
                    const µ1 = dists.beta_prime_4p.measurements.nonCentralMoments(1, alpha, beta, loc, scale);
                    const µ2 = dists.beta_prime_4p.measurements.nonCentralMoments(2, alpha, beta, loc, scale);
                    const std = Math.sqrt(µ2 - µ1 ** 2);
                    const central_µ4 = dists.beta_prime_4p.measurements.centralMoments(4, alpha, beta, loc, scale);
                    return central_µ4 / (std ** 4);
                },
                median: function (alpha, beta, loc, scale) {
                    return dists.beta_prime_4p.measurements.ppf(0.5, alpha, beta, loc, scale);
                },
                mode: function (alpha, beta, loc, scale) {
                    return loc + scale * (alpha - 1) / (beta + 1);
                },
            },
        }
    }
}
console.log(dists.beta_prime_4p.measurements.stats.mean(100, 54, 0, 20))
console.log(dists.beta_prime_4p.measurements.stats.variance(100, 54, 0, 20))
console.log(dists.beta_prime_4p.measurements.stats.standardDeviation(100, 54, 0, 20))
console.log(dists.beta_prime_4p.measurements.stats.skewness(100, 54, 0, 20))
console.log(dists.beta_prime_4p.measurements.stats.kurtosis(100, 54, 0, 20))
// console.log(dists.beta_prime_4p.measurements.stats.median(100, 54, 0, 20))
console.log(dists.beta_prime_4p.measurements.stats.mode(100, 54, 0, 20))

// mean_value: 37.73584905660374
// variance_value: 41.898293945287435
// standard_deviation_value: 6.472889149775966
// skewness_value: 0.578410671752133
// kurtosis_value: 3.6318751249615646
// median_value: 37.14265555487496
// mode_value: 36