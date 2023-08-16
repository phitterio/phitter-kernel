jStat = require("../node_modules/jstat");

dists = {
    inverse_gaussian_3p: {
        measurements: {
            nonCentralMoments: function (k, miu, lambda, loc) {
                return undefined;
            },
            centralMoments: function (k, miu, lambda, loc) {
                return undefined;
            },
            stats: {
                mean: function (miu, lambda, loc) {
                    return miu+loc;
                },
                variance: function (miu, lambda, loc) {
                    return (miu**3/lambda);
                },
                standardDeviation: function (miu, lambda, loc) {
                    return Math.sqrt(this.variance(miu, lambda, loc));
                },
                skewness: function (miu, lambda, loc) {
                    return 3*Math.sqrt(miu/lambda);
                },
                kurtosis: function (miu, lambda, loc) {
                    return 15*(miu/lambda)+3;
                },
                median: function (miu, lambda, loc) {
                    return dists.inverse_gaussian_3p.measurements.ppf(0.5, miu, lambda, loc);
                },
                mode: function (miu, lambda, loc) {
                    return loc+miu*(Math.sqrt(1+9*miu*miu/(4*lambda*lambda))-3*miu/(2*lambda));
                },
            },
        }
    }
}
console.log(dists.inverse_gaussian_3p.measurements.stats.mean(10, 100, 60))
console.log(dists.inverse_gaussian_3p.measurements.stats.variance(10, 100, 60))
console.log(dists.inverse_gaussian_3p.measurements.stats.standardDeviation(10, 100, 60))
console.log(dists.inverse_gaussian_3p.measurements.stats.skewness(10, 100, 60))
console.log(dists.inverse_gaussian_3p.measurements.stats.kurtosis(10, 100, 60))
// console.log(dists.inverse_gaussian_3p.measurements.stats.median(10, 100, 60))
console.log(dists.inverse_gaussian_3p.measurements.stats.mode(10, 100, 60))

// mean_value: 70
// variance_value: 10
// standard_deviation_value: 3.1622776601683795
// skewness_value: 0.9486832980505138
// kurtosis_value: 4.5
// median_value: None
// mode_value: 68.61187420807835