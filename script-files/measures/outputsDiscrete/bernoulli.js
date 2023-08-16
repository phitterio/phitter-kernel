jStat = require("../node_modules/jstat");

dists = {
    bernoulli: {
        measurements: {
            nonCentralMoments: function (k, p) {
                return undefined;
            },
            centralMoments: function (k, p) {
                return undefined;
            },
            stats: {
                mean: function (p) {
                    return p;
                },
                variance: function (p) {
                    return p*(1-p);
                },
                standardDeviation: function (p) {
                    return Math.sqrt(this.variance(p));
                },
                skewness: function (p) {
                    return (1-2*p)/Math.sqrt(p*(1-p));
                },
                kurtosis: function (p) {
                    return (6*p*p-6*p+1)/(p*(1-p))+3;
                },
                median: function (p) {
                    return dists.bernoulli.measurements.ppf(0.5, p);
                },
                mode: function (p) {
                    return p<0.5 ? 0 : 1;
                },
            },
        }
    }
}
console.log(dists.bernoulli.measurements.stats.mean(0.5))
console.log(dists.bernoulli.measurements.stats.variance(0.5))
console.log(dists.bernoulli.measurements.stats.standardDeviation(0.5))
console.log(dists.bernoulli.measurements.stats.skewness(0.5))
console.log(dists.bernoulli.measurements.stats.kurtosis(0.5))
// console.log(dists.bernoulli.measurements.stats.median(0.5))
console.log(dists.bernoulli.measurements.stats.mode(0.5))

// mean_value: 0.5
// variance_value: 0.25
// standard_deviation_value: 0.5
// skewness_value: 0
// kurtosis_value: 1
// median_value: 0.5
// mode_value: 1