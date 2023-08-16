jStat = require("jstat");

dists = {
    beta_prime: {
        measurements: {
            nonCentralMoments: function (k, alpha, beta) {
                return jStat.gammafn(k + alpha) * jStat.gammafn(beta - k) / (jStat.gammafn(alpha) * jStat.gammafn(beta))
            },
            centralMoments: function (k, alpha, beta) {
                const µ1 = this.nonCentralMoments(1, alpha, beta);
                const µ2 = this.nonCentralMoments(2, alpha, beta);
                const µ3 = this.nonCentralMoments(3, alpha, beta);
                const µ4 = this.nonCentralMoments(4, alpha, beta);

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
                mean: function (alpha, beta) {
                    const µ1 = dists.beta_prime.measurements.nonCentralMoments(1, alpha, beta);
                    return µ1;
                },
                variance: function (alpha, beta) {
                    const µ1 = dists.beta_prime.measurements.nonCentralMoments(1, alpha, beta);
                    const µ2 = dists.beta_prime.measurements.nonCentralMoments(2, alpha, beta);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (alpha, beta) {
                    return Math.sqrt(this.variance(alpha, beta));
                },
                skewness: function (alpha, beta) {
                    const central_µ3 = dists.beta_prime.measurements.centralMoments(3, alpha, beta);
                    return central_µ3 / (this.standardDeviation(alpha, beta) ** 3);
                },
                kurtosis: function (alpha, beta) {
                    const central_µ4 = dists.beta_prime.measurements.centralMoments(4, alpha, beta);
                    return central_µ4 / (this.standardDeviation(alpha, beta) ** 4);
                },
                median: function (alpha, beta) {
                    return dists.beta_prime.measurements.ppf(0.5, alpha, beta);
                },
                mode: function (alpha, beta) {
                    return (alpha - 1) / (beta + 1);
                },
            },
        }
    }
}
console.log(dists.beta_prime.measurements.stats.mean(100, 54))
console.log(dists.beta_prime.measurements.stats.variance(100, 54))
console.log(dists.beta_prime.measurements.stats.standardDeviation(100, 54))
console.log(dists.beta_prime.measurements.stats.skewness(100, 54))
console.log(dists.beta_prime.measurements.stats.kurtosis(100, 54))
// console.log(dists.beta_prime.measurements.stats.median(100, 54))
console.log(dists.beta_prime.measurements.stats.mode(100, 54))

// mean_value: 1.8867924528301872
// variance_value: 0.10474573486321859
// standard_deviation_value: 0.32364445748879833
// skewness_value: 0.578410671752133
// kurtosis_value: 3.6318751249615646
// median_value: 1.857132777743748
// mode_value: 1.8