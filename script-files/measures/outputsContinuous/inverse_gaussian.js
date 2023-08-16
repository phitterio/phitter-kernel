jStat = require("../node_modules/jstat");

dists = {
    inverse_gaussian: {
        measurements: {
            nonCentralMoments: function (k, miu, lambda) {
                return undefined;
            },
            centralMoments: function (k, miu, lambda) {
                return undefined;
            },
            stats: {
                mean: function (miu, lambda) {
                    return miu;
                },
                variance: function (miu, lambda) {
                    return miu**3/lambda;
                },
                standardDeviation: function (miu, lambda) {
                    return Math.sqrt(this.variance(miu, lambda));
                },
                skewness: function (miu, lambda) {
                    return 3*Math.sqrt(miu/lambda);
                },
                kurtosis: function (miu, lambda) {
                    return 15*(miu/lambda)+3;
                },
                median: function (miu, lambda) {
                    return dists.inverse_gaussian.measurements.ppf(0.5, miu, lambda);
                },
                mode: function (miu, lambda) {
                    return miu*(Math.sqrt(1+9*miu*miu/(4*lambda*lambda))-3*miu/(2*lambda));
                },
            },
        }
    }
}
console.log(dists.inverse_gaussian.measurements.stats.mean(10, 20))
console.log(dists.inverse_gaussian.measurements.stats.variance(10, 20))
console.log(dists.inverse_gaussian.measurements.stats.standardDeviation(10, 20))
console.log(dists.inverse_gaussian.measurements.stats.skewness(10, 20))
console.log(dists.inverse_gaussian.measurements.stats.kurtosis(10, 20))
// console.log(dists.inverse_gaussian.measurements.stats.median(10, 20))
console.log(dists.inverse_gaussian.measurements.stats.mode(10, 20))

// mean_value: 10
// variance_value: 50
// standard_deviation_value: 7.0710678118654755
// skewness_value: 2.121320343559643
// kurtosis_value: 10.5
// median_value: None
// mode_value: 5