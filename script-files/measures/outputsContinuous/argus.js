jStat = require("../node_modules/jstat");
BESSEL = require("../node_modules/bessel");

dists = {
    argus: {
        measurements: {
            nonCentralMoments: function (k, chi, loc, scale) {
                return undefined;
            },
            centralMoments: function (k, chi, loc, scale) {
                return undefined;
            },
            stats: {
                mean: function (chi, loc, scale) {
                    const std_cdf = (t) => 0.5 * (1 + jStat.erf(t / Math.sqrt(2)));
                    const std_pdf = (t) => (1 / Math.sqrt(2 * Math.PI)) * Math.exp(-(t ** 2 / 2));
                    return loc + scale * Math.sqrt(Math.PI / 8) * ((chi * Math.exp(-chi * chi / 4) * BESSEL.besseli(chi * chi / 4, 1)) / (std_cdf(chi) - chi * std_pdf(chi) - 0.5));
                },
                variance: function (chi, loc, scale) {
                    const std_cdf = (t) => 0.5 * (1 + jStat.erf(t / Math.sqrt(2)));
                    const std_pdf = (t) => (1 / Math.sqrt(2 * Math.PI)) * Math.exp(-(t ** 2 / 2));
                    return scale * scale * (1 - 3 / (chi * chi) + chi * std_pdf(chi) / (std_cdf(chi) - chi * std_pdf(chi) - 0.5)) - (this.mean(chi, loc, scale) - loc) ** 2;
                },
                standardDeviation: function (chi, loc, scale) {
                    return Math.sqrt(this.variance(chi, loc, scale));
                },
                skewness: function (chi, loc, scale) {
                    return undefined;
                },
                kurtosis: function (chi, loc, scale) {
                    return undefined;
                },
                median: function (chi, loc, scale) {
                    return dists.argus.measurements.ppf(0.5, chi, loc, scale);
                },
                mode: function (chi, loc, scale) {
                    return loc + scale * (1 / (Math.sqrt(2) * chi)) * Math.sqrt((chi * chi - 2) + Math.sqrt(chi * chi * chi * chi + 4));
                },
            },
        }
    }
}
console.log(dists.argus.measurements.stats.mean(0.7, 100, 9))
console.log(dists.argus.measurements.stats.variance(0.7, 100, 9))
console.log(dists.argus.measurements.stats.standardDeviation(0.7, 100, 9))
console.log(dists.argus.measurements.stats.skewness(0.7, 100, 9))
console.log(dists.argus.measurements.stats.kurtosis(0.7, 100, 9))
// console.log(dists.argus.measurements.stats.median(0.7, 100, 9))
console.log(dists.argus.measurements.stats.mode(0.7, 100, 9))

// mean_value: 105.43185648628986
// variance_value: 4.269641658445249
// standard_deviation_value: 2.0663111233416056
// skewness_value: -
// kurtosis_value: -
// median_value: 105.64444142589437
// mode_value: 106.7371324916817