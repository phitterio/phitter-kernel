const MEASUREMENTS = {
    alpha: {
        measurements: {
            nonCentralMoments: function (k, alpha, loc, scale) {
                return undefined;
            },
            centralMoments: function (k, alpha, loc, scale) {
                return undefined;
            },
            stats: {
                mean: function (alpha, loc, scale) {
                    return undefined;
                },
                variance: function (alpha, loc, scale) {
                    return undefined;
                },
                standardDeviation: function (alpha, loc, scale) {
                    if (this.variance(alpha, loc, scale) === undefined) {
                        return undefined
                    }
                    return Math.sqrt(this.variance(alpha, loc, scale));
                },
                skewness: function (alpha, loc, scale) {
                    return undefined;
                },
                kurtosis: function (alpha, loc, scale) {
                    return undefined;
                },
                median: function (alpha, loc, scale) {
                    return dists.alpha.measurements.ppf(0.5, alpha, loc, scale);
                },
                mode: function (alpha, loc, scale) {
                    return (scale * (Math.sqrt(alpha * alpha + 8) - alpha)) / 4 + loc;
                },
            },
        },
    },
    arcsine: {
        measurements: {
            nonCentralMoments: function (k, a, b) {
                return (jStat.gammafn(0.5) * jStat.gammafn(k + 0.5)) / (Math.PI * jStat.gammafn(k + 1));
            },
            centralMoments: function (k, a, b) {
                const µ1 = this.nonCentralMoments(1, a, b);
                const µ2 = this.nonCentralMoments(2, a, b);
                const µ3 = this.nonCentralMoments(3, a, b);
                const µ4 = this.nonCentralMoments(4, a, b);

                let result;
                switch (k) {
                    case 1:
                        result = 0;
                        break;
                    case 2:
                        result = µ2 - µ1 ** 2;
                        break;
                    case 3:
                        result = µ3 - 3 * µ1 * µ2 + 2 * µ1 ** 3;
                        break;
                    case 4:
                        result = µ4 - 4 * µ1 * µ3 + 6 * µ1 ** 2 * µ2 - 3 * µ1 ** 4;
                        break;
                }
                return result;
            },
            stats: {
                mean: function (a, b) {
                    const µ1 = dists.arcsine.measurements.nonCentralMoments(1, a, b);
                    return µ1 * (b - a) + a;
                },
                variance: function (a, b) {
                    const µ1 = dists.arcsine.measurements.nonCentralMoments(1, a, b);
                    const µ2 = dists.arcsine.measurements.nonCentralMoments(2, a, b);
                    return (µ2 - µ1 ** 2) * (b - a) ** 2;
                },
                standardDeviation: function (a, b) {
                    return Math.sqrt(this.variance(a, b));
                },
                skewness: function (a, b) {
                    const central_µ3 = dists.arcsine.measurements.centralMoments(3, a, b);
                    const µ1 = dists.arcsine.measurements.nonCentralMoments(1, a, b);
                    const µ2 = dists.arcsine.measurements.nonCentralMoments(2, a, b);
                    const std = Math.sqrt(µ2 - µ1 ** 2);
                    return central_µ3 / std ** 3;
                },
                kurtosis: function (a, b) {
                    const central_µ4 = dists.arcsine.measurements.centralMoments(4, a, b);
                    const µ1 = dists.arcsine.measurements.nonCentralMoments(1, a, b);
                    const µ2 = dists.arcsine.measurements.nonCentralMoments(2, a, b);
                    const std = Math.sqrt(µ2 - µ1 ** 2);
                    return central_µ4 / std ** 4;
                },
                median: function (a, b) {
                    return dists.arcsine.measurements.ppf(0.5, a, b);
                },
                mode: function (a, b) {
                    return undefined;
                },
            },
        },
    },
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
                    return (
                        loc + scale * Math.sqrt(Math.PI / 8) * ((chi * Math.exp((-chi * chi) / 4) * BESSEL.besseli((chi * chi) / 4, 1)) / (std_cdf(chi) - chi * std_pdf(chi) - 0.5))
                    );
                },
                variance: function (chi, loc, scale) {
                    const std_cdf = (t) => 0.5 * (1 + jStat.erf(t / Math.sqrt(2)));
                    const std_pdf = (t) => (1 / Math.sqrt(2 * Math.PI)) * Math.exp(-(t ** 2 / 2));
                    return scale * scale * (1 - 3 / (chi * chi) + (chi * std_pdf(chi)) / (std_cdf(chi) - chi * std_pdf(chi) - 0.5)) - (this.mean(chi, loc, scale) - loc) ** 2;
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
                    return loc + scale * (1 / (Math.sqrt(2) * chi)) * Math.sqrt(chi * chi - 2 + Math.sqrt(chi * chi * chi * chi + 4));
                },
            },
        },
    },
    beta: {
        measurements: {
            nonCentralMoments: function (k, alpha, beta, A, B) {
                return undefined;
            },
            centralMoments: function (k, alpha, beta, A, B) {
                return undefined;
            },
            stats: {
                mean: function (alpha, beta, A, B) {
                    return A + (alpha / (alpha + beta)) * (B - A);
                },
                variance: function (alpha, beta, A, B) {
                    return ((alpha * beta) / ((alpha + beta + 1) * (alpha + beta) ** 2)) * (B - A) ** 2;
                },
                standardDeviation: function (alpha, beta, A, B) {
                    return Math.sqrt(this.variance(alpha, beta, A, B));
                },
                skewness: function (alpha, beta, A, B) {
                    return 2 * ((beta - alpha) / (alpha + beta + 2)) * Math.sqrt((alpha + beta + 1) / (alpha * beta));
                },
                kurtosis: function (alpha, beta, A, B) {
                    return 3 + (6 * ((alpha + beta + 1) * (alpha - beta) ** 2 - alpha * beta * (alpha + beta + 2))) / (alpha * beta * (alpha + beta + 2) * (alpha + beta + 3));
                },
                median: function (alpha, beta, A, B) {
                    return dists.beta.measurements.ppf(0.5, alpha, beta, A, B);
                },
                mode: function (alpha, beta, A, B) {
                    return A + ((alpha - 1) / (alpha + beta - 2)) * (B - A);
                },
            },
        },
    },
    beta_prime: {
        measurements: {
            nonCentralMoments: function (k, alpha, beta) {
                return (jStat.gammafn(k + alpha) * jStat.gammafn(beta - k)) / (jStat.gammafn(alpha) * jStat.gammafn(beta));
            },
            centralMoments: function (k, alpha, beta) {
                const µ1 = this.nonCentralMoments(1, alpha, beta);
                const µ2 = this.nonCentralMoments(2, alpha, beta);
                const µ3 = this.nonCentralMoments(3, alpha, beta);
                const µ4 = this.nonCentralMoments(4, alpha, beta);

                let result;
                switch (k) {
                    case 1:
                        result = 0;
                        break;
                    case 2:
                        result = µ2 - µ1 ** 2;
                        break;
                    case 3:
                        result = µ3 - 3 * µ1 * µ2 + 2 * µ1 ** 3;
                        break;
                    case 4:
                        result = µ4 - 4 * µ1 * µ3 + 6 * µ1 ** 2 * µ2 - 3 * µ1 ** 4;
                        break;
                }
                return result;
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
                    return central_µ3 / this.standardDeviation(alpha, beta) ** 3;
                },
                kurtosis: function (alpha, beta) {
                    const central_µ4 = dists.beta_prime.measurements.centralMoments(4, alpha, beta);
                    return central_µ4 / this.standardDeviation(alpha, beta) ** 4;
                },
                median: function (alpha, beta) {
                    return dists.beta_prime.measurements.ppf(0.5, alpha, beta);
                },
                mode: function (alpha, beta) {
                    return (alpha - 1) / (beta + 1);
                },
            },
        },
    },
    beta_prime_4p: {
        measurements: {
            nonCentralMoments: function (k, alpha, beta, loc, scale) {
                return (jStat.gammafn(k + alpha) * jStat.gammafn(beta - k)) / (jStat.gammafn(alpha) * jStat.gammafn(beta));
            },
            centralMoments: function (k, alpha, beta, loc, scale) {
                const µ1 = this.nonCentralMoments(1, alpha, beta, loc, scale);
                const µ2 = this.nonCentralMoments(2, alpha, beta, loc, scale);
                const µ3 = this.nonCentralMoments(3, alpha, beta, loc, scale);
                const µ4 = this.nonCentralMoments(4, alpha, beta, loc, scale);

                let result;
                switch (k) {
                    case 1:
                        result = 0;
                        break;
                    case 2:
                        result = µ2 - µ1 ** 2;
                        break;
                    case 3:
                        result = µ3 - 3 * µ1 * µ2 + 2 * µ1 ** 3;
                        break;
                    case 4:
                        result = µ4 - 4 * µ1 * µ3 + 6 * µ1 ** 2 * µ2 - 3 * µ1 ** 4;
                        break;
                }
                return result;
            },
            stats: {
                mean: function (alpha, beta, loc, scale) {
                    const µ1 = dists.beta_prime_4p.measurements.nonCentralMoments(1, alpha, beta, loc, scale);
                    return loc + scale * µ1;
                },
                variance: function (alpha, beta, loc, scale) {
                    const µ1 = dists.beta_prime_4p.measurements.nonCentralMoments(1, alpha, beta, loc, scale);
                    const µ2 = dists.beta_prime_4p.measurements.nonCentralMoments(2, alpha, beta, loc, scale);
                    return scale ** 2 * (µ2 - µ1 ** 2);
                },
                standardDeviation: function (alpha, beta, loc, scale) {
                    return Math.sqrt(this.variance(alpha, beta, loc, scale));
                },
                skewness: function (alpha, beta, loc, scale) {
                    const µ1 = dists.beta_prime_4p.measurements.nonCentralMoments(1, alpha, beta, loc, scale);
                    const µ2 = dists.beta_prime_4p.measurements.nonCentralMoments(2, alpha, beta, loc, scale);
                    const std = Math.sqrt(µ2 - µ1 ** 2);
                    const central_µ3 = dists.beta_prime_4p.measurements.centralMoments(3, alpha, beta, loc, scale);
                    return central_µ3 / std ** 3;
                },
                kurtosis: function (alpha, beta, loc, scale) {
                    const µ1 = dists.beta_prime_4p.measurements.nonCentralMoments(1, alpha, beta, loc, scale);
                    const µ2 = dists.beta_prime_4p.measurements.nonCentralMoments(2, alpha, beta, loc, scale);
                    const std = Math.sqrt(µ2 - µ1 ** 2);
                    const central_µ4 = dists.beta_prime_4p.measurements.centralMoments(4, alpha, beta, loc, scale);
                    return central_µ4 / std ** 4;
                },
                median: function (alpha, beta, loc, scale) {
                    return dists.beta_prime_4p.measurements.ppf(0.5, alpha, beta, loc, scale);
                },
                mode: function (alpha, beta, loc, scale) {
                    return loc + (scale * (alpha - 1)) / (beta + 1);
                },
            },
        },
    },
    bradford: {
        measurements: {
            nonCentralMoments: function (k, c, min, max) {
                return undefined;
            },
            centralMoments: function (k, c, min, max) {
                return undefined;
            },
            stats: {
                mean: function (c, min, max) {
                    return (c * (max - min) + Math.log(1 + c) * (min * (c + 1) - max)) / (Math.log(1 + c) * c);
                },
                variance: function (c, min, max) {
                    return ((max - min) ** 2 * ((c + 2) * Math.log(1 + c) - 2 * c)) / (2 * c * Math.log(1 + c) ** 2);
                },
                standardDeviation: function (c, min, max) {
                    return Math.sqrt(this.variance(c, min, max));
                },
                skewness: function (c, min, max) {
                    return (
                        (Math.sqrt(2) * (12 * c * c - 9 * Math.log(1 + c) * c * (c + 2) + 2 * Math.log(1 + c) * Math.log(1 + c) * (c * (c + 3) + 3))) /
                        (Math.sqrt(c * (c * (Math.log(1 + c) - 2) + 2 * Math.log(1 + c))) * (3 * c * (Math.log(1 + c) - 2) + 6 * Math.log(1 + c)))
                    );
                },
                kurtosis: function (c, min, max) {
                    return (
                        (c ** 3 * (Math.log(1 + c) - 3) * (Math.log(1 + c) * (3 * Math.log(1 + c) - 16) + 24) +
                            12 * Math.log(1 + c) * c * c * (Math.log(1 + c) - 4) * (Math.log(1 + c) - 3) +
                            6 * c * Math.log(1 + c) ** 2 * (3 * Math.log(1 + c) - 14) +
                            12 * Math.log(1 + c) ** 3) /
                            (3 * c * (c * (Math.log(1 + c) - 2) + 2 * Math.log(1 + c)) ** 2) +
                        3
                    );
                },
                median: function (c, min, max) {
                    return dists.bradford.measurements.ppf(0.5, c, min, max);
                },
                mode: function (c, min, max) {
                    return min;
                },
            },
        },
    },
    burr: {
        measurements: {
            nonCentralMoments: function (k, A, B, C) {
                return A ** k * C * ((jStat.gammafn((B * C - k) / B) * jStat.gammafn((B + k) / B)) / jStat.gammafn((B * C - k) / B + (B + k) / B));
            },
            centralMoments: function (k, A, B, C) {
                const µ1 = this.nonCentralMoments(1, A, B, C);
                const µ2 = this.nonCentralMoments(2, A, B, C);
                const µ3 = this.nonCentralMoments(3, A, B, C);
                const µ4 = this.nonCentralMoments(4, A, B, C);

                let result;
                switch (k) {
                    case 1:
                        result = 0;
                        break;
                    case 2:
                        result = µ2 - µ1 ** 2;
                        break;
                    case 3:
                        result = µ3 - 3 * µ1 * µ2 + 2 * µ1 ** 3;
                        break;
                    case 4:
                        result = µ4 - 4 * µ1 * µ3 + 6 * µ1 ** 2 * µ2 - 3 * µ1 ** 4;
                        break;
                }
                return result;
            },
            stats: {
                mean: function (A, B, C) {
                    const µ1 = dists.burr.measurements.nonCentralMoments(1, A, B, C);
                    return µ1;
                },
                variance: function (A, B, C) {
                    const µ1 = dists.burr.measurements.nonCentralMoments(1, A, B, C);
                    const µ2 = dists.burr.measurements.nonCentralMoments(2, A, B, C);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (A, B, C) {
                    return Math.sqrt(this.variance(A, B, C));
                },
                skewness: function (A, B, C) {
                    const central_µ3 = dists.burr.measurements.centralMoments(3, A, B, C);
                    return central_µ3 / this.standardDeviation(A, B, C) ** 3;
                },
                kurtosis: function (A, B, C) {
                    const central_µ4 = dists.burr.measurements.centralMoments(4, A, B, C);
                    return central_µ4 / this.standardDeviation(A, B, C) ** 4;
                },
                median: function (A, B, C) {
                    return dists.burr.measurements.ppf(0.5, A, B, C);
                },
                mode: function (A, B, C) {
                    return A * ((B - 1) / (B * C + 1)) ** (1 / B);
                },
            },
        },
    },
    burr_4p: {
        measurements: {
            nonCentralMoments: function (k, A, B, C, loc) {
                return A ** k * C * ((jStat.gammafn((B * C - k) / B) * jStat.gammafn((B + k) / B)) / jStat.gammafn((B * C - k) / B + (B + k) / B));
            },
            centralMoments: function (k, A, B, C, loc) {
                const µ1 = this.nonCentralMoments(1, A, B, C, loc);
                const µ2 = this.nonCentralMoments(2, A, B, C, loc);
                const µ3 = this.nonCentralMoments(3, A, B, C, loc);
                const µ4 = this.nonCentralMoments(4, A, B, C, loc);

                let result;
                switch (k) {
                    case 1:
                        result = 0;
                        break;
                    case 2:
                        result = µ2 - µ1 ** 2;
                        break;
                    case 3:
                        result = µ3 - 3 * µ1 * µ2 + 2 * µ1 ** 3;
                        break;
                    case 4:
                        result = µ4 - 4 * µ1 * µ3 + 6 * µ1 ** 2 * µ2 - 3 * µ1 ** 4;
                        break;
                }
                return result;
            },
            stats: {
                mean: function (A, B, C, loc) {
                    const µ1 = dists.burr_4p.measurements.nonCentralMoments(1, A, B, C, loc);
                    return loc + µ1;
                },
                variance: function (A, B, C, loc) {
                    const µ1 = dists.burr_4p.measurements.nonCentralMoments(1, A, B, C, loc);
                    const µ2 = dists.burr_4p.measurements.nonCentralMoments(2, A, B, C, loc);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (A, B, C, loc) {
                    return Math.sqrt(this.variance(A, B, C, loc));
                },
                skewness: function (A, B, C, loc) {
                    const central_µ3 = dists.burr_4p.measurements.centralMoments(3, A, B, C, loc);
                    return central_µ3 / this.standardDeviation(A, B, C, loc) ** 3;
                },
                kurtosis: function (A, B, C, loc) {
                    const central_µ4 = dists.burr_4p.measurements.centralMoments(4, A, B, C, loc);
                    return central_µ4 / this.standardDeviation(A, B, C, loc) ** 4;
                },
                median: function (A, B, C, loc) {
                    return dists.burr_4p.measurements.ppf(0.5, A, B, C, loc);
                },
                mode: function (A, B, C, loc) {
                    return loc + A * ((B - 1) / (B * C + 1)) ** (1 / B);
                },
            },
        },
    },
    cauchy: {
        measurements: {
            nonCentralMoments: function (k, x0, gamma) {
                return undefined;
            },
            centralMoments: function (k, x0, gamma) {
                return undefined;
            },
            stats: {
                mean: function (x0, gamma) {
                    return undefined;
                },
                variance: function (x0, gamma) {
                    return undefined;
                },
                standardDeviation: function (x0, gamma) {
                    return Math.sqrt(this.variance(x0, gamma));
                },
                skewness: function (x0, gamma) {
                    return undefined;
                },
                kurtosis: function (x0, gamma) {
                    return undefined;
                },
                median: function (x0, gamma) {
                    return dists.cauchy.measurements.ppf(0.5, x0, gamma);
                },
                mode: function (x0, gamma) {
                    return x0;
                },
            },
        },
    },
    chi_square: {
        measurements: {
            nonCentralMoments: function (k, df) {
                return undefined;
            },
            centralMoments: function (k, df) {
                return undefined;
            },
            stats: {
                mean: function (df) {
                    return df;
                },
                variance: function (df) {
                    return df * 2;
                },
                standardDeviation: function (df) {
                    return Math.sqrt(this.variance(df));
                },
                skewness: function (df) {
                    return Math.sqrt(8 / df);
                },
                kurtosis: function (df) {
                    return 12 / df + 3;
                },
                median: function (df) {
                    return dists.chi_square.measurements.ppf(0.5, df);
                },
                mode: function (df) {
                    return df - 2;
                },
            },
        },
    },
    chi_square_3p: {
        measurements: {
            nonCentralMoments: function (k, df, loc, scale) {
                return undefined;
            },
            centralMoments: function (k, df, loc, scale) {
                return undefined;
            },
            stats: {
                mean: function (df, loc, scale) {
                    return df * scale + loc;
                },
                variance: function (df, loc, scale) {
                    return df * 2 * (scale * scale);
                },
                standardDeviation: function (df, loc, scale) {
                    return Math.sqrt(this.variance(df, loc, scale));
                },
                skewness: function (df, loc, scale) {
                    return Math.sqrt(8 / df);
                },
                kurtosis: function (df, loc, scale) {
                    return 12 / df + 3;
                },
                median: function (df, loc, scale) {
                    return dists.chi_square_3p.measurements.ppf(0.5, df, loc, scale);
                },
                mode: function (df, loc, scale) {
                    return (df - 2) * scale + loc;
                },
            },
        },
    },
    dagum: {
        measurements: {
            nonCentralMoments: function (k, a, b, p) {
                return b ** k * p * ((jStat.gammafn((a * p + k) / a) * jStat.gammafn((a - k) / a)) / jStat.gammafn((a * p + k) / a + (a - k) / a));
            },
            centralMoments: function (k, a, b, p) {
                const µ1 = this.nonCentralMoments(1, a, b, p);
                const µ2 = this.nonCentralMoments(2, a, b, p);
                const µ3 = this.nonCentralMoments(3, a, b, p);
                const µ4 = this.nonCentralMoments(4, a, b, p);

                let result;
                switch (k) {
                    case 1:
                        result = 0;
                        break;
                    case 2:
                        result = µ2 - µ1 ** 2;
                        break;
                    case 3:
                        result = µ3 - 3 * µ1 * µ2 + 2 * µ1 ** 3;
                        break;
                    case 4:
                        result = µ4 - 4 * µ1 * µ3 + 6 * µ1 ** 2 * µ2 - 3 * µ1 ** 4;
                        break;
                }
                return result;
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
                    return central_µ3 / this.standardDeviation(a, b, p) ** 3;
                },
                kurtosis: function (a, b, p) {
                    const central_µ4 = dists.dagum.measurements.centralMoments(4, a, b, p);
                    return central_µ4 / this.standardDeviation(a, b, p) ** 4;
                },
                median: function (a, b, p) {
                    return dists.dagum.measurements.ppf(0.5, a, b, p);
                },
                mode: function (a, b, p) {
                    return b * ((a * p - 1) / (a + 1)) ** (1 / a);
                },
            },
        },
    },
    dagum_4p: {
        measurements: {
            nonCentralMoments: function (k, a, b, p, loc) {
                return b ** k * p * ((jStat.gammafn((a * p + k) / a) * jStat.gammafn((a - k) / a)) / jStat.gammafn((a * p + k) / a + (a - k) / a));
            },
            centralMoments: function (k, a, b, p, loc) {
                const µ1 = this.nonCentralMoments(1, a, b, p, loc);
                const µ2 = this.nonCentralMoments(2, a, b, p, loc);
                const µ3 = this.nonCentralMoments(3, a, b, p, loc);
                const µ4 = this.nonCentralMoments(4, a, b, p, loc);

                let result;
                switch (k) {
                    case 1:
                        result = 0;
                        break;
                    case 2:
                        result = µ2 - µ1 ** 2;
                        break;
                    case 3:
                        result = µ3 - 3 * µ1 * µ2 + 2 * µ1 ** 3;
                        break;
                    case 4:
                        result = µ4 - 4 * µ1 * µ3 + 6 * µ1 ** 2 * µ2 - 3 * µ1 ** 4;
                        break;
                }
                return result;
            },
            stats: {
                mean: function (a, b, p, loc) {
                    const µ1 = dists.dagum_4p.measurements.nonCentralMoments(1, a, b, p, loc);
                    return loc + µ1;
                },
                variance: function (a, b, p, loc) {
                    const µ1 = dists.dagum_4p.measurements.nonCentralMoments(1, a, b, p, loc);
                    const µ2 = dists.dagum_4p.measurements.nonCentralMoments(2, a, b, p, loc);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (a, b, p, loc) {
                    return Math.sqrt(this.variance(a, b, p, loc));
                },
                skewness: function (a, b, p, loc) {
                    const central_µ3 = dists.dagum_4p.measurements.centralMoments(3, a, b, p, loc);
                    return central_µ3 / this.standardDeviation(a, b, p, loc) ** 3;
                },
                kurtosis: function (a, b, p, loc) {
                    const central_µ4 = dists.dagum_4p.measurements.centralMoments(4, a, b, p, loc);
                    return central_µ4 / this.standardDeviation(a, b, p, loc) ** 4;
                },
                median: function (a, b, p, loc) {
                    return dists.dagum_4p.measurements.ppf(0.5, a, b, p, loc);
                },
                mode: function (a, b, p, loc) {
                    return loc + b * ((a * p - 1) / (a + 1)) ** (1 / a);
                },
            },
        },
    },
    erlang: {
        measurements: {
            nonCentralMoments: function (k, k_, beta) {
                return beta ** k * (jStat.gammafn(k + k_) / jStat.factorial(k_ - 1));
            },
            centralMoments: function (k, k, beta) {
                const µ1 = this.nonCentralMoments(1, k, beta);
                const µ2 = this.nonCentralMoments(2, k, beta);
                const µ3 = this.nonCentralMoments(3, k, beta);
                const µ4 = this.nonCentralMoments(4, k, beta);

                let result;
                switch (k) {
                    case 1:
                        result = 0;
                        break;
                    case 2:
                        result = µ2 - µ1 ** 2;
                        break;
                    case 3:
                        result = µ3 - 3 * µ1 * µ2 + 2 * µ1 ** 3;
                        break;
                    case 4:
                        result = µ4 - 4 * µ1 * µ3 + 6 * µ1 ** 2 * µ2 - 3 * µ1 ** 4;
                        break;
                }
                return result;
            },
            stats: {
                mean: function (k, beta) {
                    const µ1 = dists.erlang.measurements.nonCentralMoments(1, k, beta);
                    return µ1;
                },
                variance: function (k, beta) {
                    const µ1 = dists.erlang.measurements.nonCentralMoments(1, k, beta);
                    const µ2 = dists.erlang.measurements.nonCentralMoments(2, k, beta);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (k, beta) {
                    return Math.sqrt(this.variance(k, beta));
                },
                skewness: function (k, beta) {
                    const central_µ3 = dists.erlang.measurements.centralMoments(3, k, beta);
                    return central_µ3 / this.standardDeviation(k, beta) ** 3;
                },
                kurtosis: function (k, beta) {
                    const central_µ4 = dists.erlang.measurements.centralMoments(4, k, beta);
                    return central_µ4 / this.standardDeviation(k, beta) ** 4;
                },
                median: function (k, beta) {
                    return dists.erlang.measurements.ppf(0.5, k, beta);
                },
                mode: function (k, beta) {
                    return beta * (k - 1);
                },
            },
        },
    },
    erlang_3p: {
        measurements: {
            nonCentralMoments: function (k, k_, beta, loc) {
                return beta ** k * (jStat.gammafn(k_ + k) / jStat.factorial(k_ - 1));
            },
            centralMoments: function (k, k_, beta, loc) {
                const µ1 = this.nonCentralMoments(1, k_, beta, loc);
                const µ2 = this.nonCentralMoments(2, k_, beta, loc);
                const µ3 = this.nonCentralMoments(3, k_, beta, loc);
                const µ4 = this.nonCentralMoments(4, k_, beta, loc);

                let result;
                switch (k) {
                    case 1:
                        result = 0;
                        break;
                    case 2:
                        result = µ2 - µ1 ** 2;
                        break;
                    case 3:
                        result = µ3 - 3 * µ1 * µ2 + 2 * µ1 ** 3;
                        break;
                    case 4:
                        result = µ4 - 4 * µ1 * µ3 + 6 * µ1 ** 2 * µ2 - 3 * µ1 ** 4;
                        break;
                }
                return result;
            },
            stats: {
                mean: function (k, beta, loc) {
                    const µ1 = dists.erlang_3p.measurements.nonCentralMoments(1, k, beta, loc);
                    return loc + µ1;
                },
                variance: function (k, beta, loc) {
                    const µ1 = dists.erlang_3p.measurements.nonCentralMoments(1, k, beta, loc);
                    const µ2 = dists.erlang_3p.measurements.nonCentralMoments(2, k, beta, loc);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (k, beta, loc) {
                    return Math.sqrt(this.variance(k, beta, loc));
                },
                skewness: function (k, beta, loc) {
                    const central_µ3 = dists.erlang_3p.measurements.centralMoments(3, k, beta, loc);
                    return central_µ3 / this.standardDeviation(k, beta, loc) ** 3;
                },
                kurtosis: function (k, beta, loc) {
                    const central_µ4 = dists.erlang_3p.measurements.centralMoments(4, k, beta, loc);
                    return central_µ4 / this.standardDeviation(k, beta, loc) ** 4;
                },
                median: function (k, beta, loc) {
                    return dists.erlang_3p.measurements.ppf(0.5, k, beta, loc);
                },
                mode: function (k, beta, loc) {
                    return beta * (k - 1) + loc;
                },
            },
        },
    },
    error_function: {
        measurements: {
            nonCentralMoments: function (k, h) {
                return undefined;
            },
            centralMoments: function (k, h) {
                return undefined;
            },
            stats: {
                mean: function (h) {
                    return 0;
                },
                variance: function (h) {
                    return 1 / (2 * h ** 2);
                },
                standardDeviation: function (h) {
                    return Math.sqrt(this.variance(h));
                },
                skewness: function (h) {
                    return 0;
                },
                kurtosis: function (h) {
                    return 3;
                },
                median: function (h) {
                    return dists.error_function.measurements.ppf(0.5, h);
                },
                mode: function (h) {
                    return 0;
                },
            },
        },
    },
    exponential: {
        measurements: {
            nonCentralMoments: function (k, lambda) {
                return undefined;
            },
            centralMoments: function (k, lambda) {
                return undefined;
            },
            stats: {
                mean: function (lambda) {
                    return 1 / lambda;
                },
                variance: function (lambda) {
                    return 1 / (lambda * lambda);
                },
                standardDeviation: function (lambda) {
                    return Math.sqrt(this.variance(lambda));
                },
                skewness: function (lambda) {
                    return 2;
                },
                kurtosis: function (lambda) {
                    return 9;
                },
                median: function (lambda) {
                    return dists.exponential.measurements.ppf(0.5, lambda);
                },
                mode: function (lambda) {
                    return 0;
                },
            },
        },
    },
    exponential_2p: {
        measurements: {
            nonCentralMoments: function (k, lambda, loc) {
                return undefined;
            },
            centralMoments: function (k, lambda, loc) {
                return undefined;
            },
            stats: {
                mean: function (lambda, loc) {
                    return 1 / lambda + loc;
                },
                variance: function (lambda, loc) {
                    return 1 / (lambda * lambda);
                },
                standardDeviation: function (lambda, loc) {
                    return Math.sqrt(this.variance(lambda, loc));
                },
                skewness: function (lambda, loc) {
                    return 2;
                },
                kurtosis: function (lambda, loc) {
                    return 9;
                },
                median: function (lambda, loc) {
                    return dists.exponential_2p.measurements.ppf(0.5, lambda, loc);
                },
                mode: function (lambda, loc) {
                    return loc;
                },
            },
        },
    },
    f: {
        measurements: {
            nonCentralMoments: function (k, df1, df2) {
                return (df2 / df1) ** k * (jStat.gammafn(df1 / 2 + k) / jStat.gammafn(df1 / 2)) * (jStat.gammafn(df2 / 2 - k) / jStat.gammafn(df2 / 2));
            },
            centralMoments: function (k, df1, df2) {
                const µ1 = this.nonCentralMoments(1, df1, df2);
                const µ2 = this.nonCentralMoments(2, df1, df2);
                const µ3 = this.nonCentralMoments(3, df1, df2);
                const µ4 = this.nonCentralMoments(4, df1, df2);

                let result;
                switch (k) {
                    case 1:
                        result = 0;
                        break;
                    case 2:
                        result = µ2 - µ1 ** 2;
                        break;
                    case 3:
                        result = µ3 - 3 * µ1 * µ2 + 2 * µ1 ** 3;
                        break;
                    case 4:
                        result = µ4 - 4 * µ1 * µ3 + 6 * µ1 ** 2 * µ2 - 3 * µ1 ** 4;
                        break;
                }
                return result;
            },
            stats: {
                mean: function (df1, df2) {
                    const µ1 = dists.f.measurements.nonCentralMoments(1, df1, df2);
                    return µ1;
                },
                variance: function (df1, df2) {
                    const µ1 = dists.f.measurements.nonCentralMoments(1, df1, df2);
                    const µ2 = dists.f.measurements.nonCentralMoments(2, df1, df2);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (df1, df2) {
                    return Math.sqrt(this.variance(df1, df2));
                },
                skewness: function (df1, df2) {
                    const central_µ3 = dists.f.measurements.centralMoments(3, df1, df2);
                    return central_µ3 / this.standardDeviation(df1, df2) ** 3;
                },
                kurtosis: function (df1, df2) {
                    const central_µ4 = dists.f.measurements.centralMoments(4, df1, df2);
                    return central_µ4 / this.standardDeviation(df1, df2) ** 4;
                },
                median: function (df1, df2) {
                    return dists.f.measurements.ppf(0.5, df1, df2);
                },
                mode: function (df1, df2) {
                    return (df2 * (df1 - 2)) / (df1 * (df2 + 2));
                },
            },
        },
    },
    fatigue_life: {
        measurements: {
            nonCentralMoments: function (k, gamma, loc, scale) {
                return undefined;
            },
            centralMoments: function (k, gamma, loc, scale) {
                return undefined;
            },
            stats: {
                mean: function (gamma, loc, scale) {
                    return loc + scale * (1 + gamma ** 2 / 2);
                },
                variance: function (gamma, loc, scale) {
                    return scale ** 2 * gamma ** 2 * (1 + (5 * gamma ** 2) / 4);
                },
                standardDeviation: function (gamma, loc, scale) {
                    return Math.sqrt(this.variance(gamma, loc, scale));
                },
                skewness: function (gamma, loc, scale) {
                    return (4 * gamma ** 2 * (11 * gamma ** 2 + 6)) / ((5 * gamma ** 2 + 4) * Math.sqrt(gamma ** 2 * (5 * gamma ** 2 + 4)));
                },
                kurtosis: function (gamma, loc, scale) {
                    return 3 + (6 * gamma * gamma * (93 * gamma * gamma + 40)) / (5 * gamma ** 2 + 4) ** 2;
                },
                median: function (gamma, loc, scale) {
                    return dists.fatigue_life.measurements.ppf(0.5, gamma, loc, scale);
                },
                mode: function (gamma, loc, scale) {
                    return undefined;
                },
            },
        },
    },
    folded_normal: {
        measurements: {
            nonCentralMoments: function (k, miu, sigma) {
                return undefined;
            },
            centralMoments: function (k, miu, sigma) {
                return undefined;
            },
            stats: {
                mean: function (miu, sigma) {
                    const std_cdf = (t) => 0.5 * (1 + jStat.erf(t / Math.sqrt(2)));
                    return sigma * Math.sqrt(2 / Math.PI) * Math.exp((-miu * miu) / (2 * sigma * sigma)) - miu * (2 * std_cdf(-miu / sigma) - 1);
                },
                variance: function (miu, sigma) {
                    return miu * miu + sigma * sigma - this.mean(miu, sigma) ** 2;
                },
                standardDeviation: function (miu, sigma) {
                    return Math.sqrt(this.variance(miu, sigma));
                },
                skewness: function (miu, sigma) {
                    return undefined;
                },
                kurtosis: function (miu, sigma) {
                    return undefined;
                },
                median: function (miu, sigma) {
                    return dists.folded_normal.measurements.ppf(0.5, miu, sigma);
                },
                mode: function (miu, sigma) {
                    return miu;
                },
            },
        },
    },
    frechet: {
        measurements: {
            nonCentralMoments: function (k, alpha, loc, scale) {
                return jStat.gammafn(1 - k / alpha);
            },
            centralMoments: function (k, alpha, loc, scale) {
                const µ1 = this.nonCentralMoments(1, alpha, loc, scale);
                const µ2 = this.nonCentralMoments(2, alpha, loc, scale);
                const µ3 = this.nonCentralMoments(3, alpha, loc, scale);
                const µ4 = this.nonCentralMoments(4, alpha, loc, scale);

                let result;
                switch (k) {
                    case 1:
                        result = 0;
                        break;
                    case 2:
                        result = µ2 - µ1 ** 2;
                        break;
                    case 3:
                        result = µ3 - 3 * µ1 * µ2 + 2 * µ1 ** 3;
                        break;
                    case 4:
                        result = µ4 - 4 * µ1 * µ3 + 6 * µ1 ** 2 * µ2 - 3 * µ1 ** 4;
                        break;
                }
                return result;
            },
            stats: {
                mean: function (alpha, loc, scale) {
                    const µ1 = dists.frechet.measurements.nonCentralMoments(1, alpha, loc, scale);
                    return loc + scale * µ1;
                },
                variance: function (alpha, loc, scale) {
                    const µ1 = dists.frechet.measurements.nonCentralMoments(1, alpha, loc, scale);
                    const µ2 = dists.frechet.measurements.nonCentralMoments(2, alpha, loc, scale);
                    return scale ** 2 * (µ2 - µ1 ** 2);
                },
                standardDeviation: function (alpha, loc, scale) {
                    return Math.sqrt(this.variance(alpha, loc, scale));
                },
                skewness: function (alpha, loc, scale) {
                    const central_µ3 = dists.frechet.measurements.centralMoments(3, alpha, loc, scale);
                    const µ1 = dists.frechet.measurements.nonCentralMoments(1, alpha, loc, scale);
                    const µ2 = dists.frechet.measurements.nonCentralMoments(2, alpha, loc, scale);
                    const std = Math.sqrt(µ2 - µ1 ** 2);
                    return central_µ3 / std ** 3;
                },
                kurtosis: function (alpha, loc, scale) {
                    const central_µ4 = dists.frechet.measurements.centralMoments(4, alpha, loc, scale);
                    const µ1 = dists.frechet.measurements.nonCentralMoments(1, alpha, loc, scale);
                    const µ2 = dists.frechet.measurements.nonCentralMoments(2, alpha, loc, scale);
                    const std = Math.sqrt(µ2 - µ1 ** 2);
                    return central_µ4 / std ** 4;
                },
                median: function (alpha, loc, scale) {
                    return dists.frechet.measurements.ppf(0.5, alpha, loc, scale);
                },
                mode: function (alpha, loc, scale) {
                    return loc + scale * (alpha / (alpha + 1)) ** (1 / alpha);
                },
            },
        },
    },
    f_4p: {
        measurements: {
            nonCentralMoments: function (k, df1, df2, loc, scale) {
                return (df2 / df1) ** k * (jStat.gammafn(df1 / 2 + k) / jStat.gammafn(df1 / 2)) * (jStat.gammafn(df2 / 2 - k) / jStat.gammafn(df2 / 2));
            },
            centralMoments: function (k, df1, df2, loc, scale) {
                const µ1 = this.nonCentralMoments(1, df1, df2, loc, scale);
                const µ2 = this.nonCentralMoments(2, df1, df2, loc, scale);
                const µ3 = this.nonCentralMoments(3, df1, df2, loc, scale);
                const µ4 = this.nonCentralMoments(4, df1, df2, loc, scale);

                let result;
                switch (k) {
                    case 1:
                        result = 0;
                        break;
                    case 2:
                        result = µ2 - µ1 ** 2;
                        break;
                    case 3:
                        result = µ3 - 3 * µ1 * µ2 + 2 * µ1 ** 3;
                        break;
                    case 4:
                        result = µ4 - 4 * µ1 * µ3 + 6 * µ1 ** 2 * µ2 - 3 * µ1 ** 4;
                        break;
                }
                return result;
            },
            stats: {
                mean: function (df1, df2, loc, scale) {
                    const µ1 = dists.f_4p.measurements.nonCentralMoments(1, df1, df2, loc, scale);
                    return loc + scale * µ1;
                },
                variance: function (df1, df2, loc, scale) {
                    const µ1 = dists.f_4p.measurements.nonCentralMoments(1, df1, df2, loc, scale);
                    const µ2 = dists.f_4p.measurements.nonCentralMoments(2, df1, df2, loc, scale);
                    return scale ** 2 * (µ2 - µ1 ** 2);
                },
                standardDeviation: function (df1, df2, loc, scale) {
                    return Math.sqrt(this.variance(df1, df2, loc, scale));
                },
                skewness: function (df1, df2, loc, scale) {
                    const central_µ3 = dists.f_4p.measurements.centralMoments(3, df1, df2, loc, scale);
                    return central_µ3 / this.standardDeviation(df1, df2, loc, scale) ** 3;
                },
                kurtosis: function (df1, df2, loc, scale) {
                    const central_µ4 = dists.f_4p.measurements.centralMoments(4, df1, df2, loc, scale);
                    return central_µ4 / this.standardDeviation(df1, df2, loc, scale) ** 4;
                },
                median: function (df1, df2, loc, scale) {
                    return dists.f_4p.measurements.ppf(0.5, df1, df2, loc, scale);
                },
                mode: function (df1, df2, loc, scale) {
                    return ((df2 * (df1 - 2)) / (df1 * (df2 + 2))) * scale + loc;
                },
            },
        },
    },
    gamma: {
        measurements: {
            nonCentralMoments: function (k, alpha, beta) {
                return beta ** k * (jStat.gammafn(k + alpha) / jStat.gammafn(alpha));
            },
            centralMoments: function (k, alpha, beta) {
                const µ1 = this.nonCentralMoments(1, alpha, beta);
                const µ2 = this.nonCentralMoments(2, alpha, beta);
                const µ3 = this.nonCentralMoments(3, alpha, beta);
                const µ4 = this.nonCentralMoments(4, alpha, beta);

                let result;
                switch (k) {
                    case 1:
                        result = 0;
                        break;
                    case 2:
                        result = µ2 - µ1 ** 2;
                        break;
                    case 3:
                        result = µ3 - 3 * µ1 * µ2 + 2 * µ1 ** 3;
                        break;
                    case 4:
                        result = µ4 - 4 * µ1 * µ3 + 6 * µ1 ** 2 * µ2 - 3 * µ1 ** 4;
                        break;
                }
                return result;
            },
            stats: {
                mean: function (alpha, beta) {
                    const µ1 = dists.gamma.measurements.nonCentralMoments(1, alpha, beta);
                    return µ1;
                },
                variance: function (alpha, beta) {
                    const µ1 = dists.gamma.measurements.nonCentralMoments(1, alpha, beta);
                    const µ2 = dists.gamma.measurements.nonCentralMoments(2, alpha, beta);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (alpha, beta) {
                    return Math.sqrt(this.variance(alpha, beta));
                },
                skewness: function (alpha, beta) {
                    const central_µ3 = dists.gamma.measurements.centralMoments(3, alpha, beta);
                    return central_µ3 / this.standardDeviation(alpha, beta) ** 3;
                },
                kurtosis: function (alpha, beta) {
                    const central_µ4 = dists.gamma.measurements.centralMoments(4, alpha, beta);
                    return central_µ4 / this.standardDeviation(alpha, beta) ** 4;
                },
                median: function (alpha, beta) {
                    return dists.gamma.measurements.ppf(0.5, alpha, beta);
                },
                mode: function (alpha, beta) {
                    return beta * (alpha - 1);
                },
            },
        },
    },
    gamma_3p: {
        measurements: {
            nonCentralMoments: function (k, alpha, beta, loc) {
                return beta ** k * (jStat.gammafn(k + alpha) / jStat.factorial(alpha - 1));
            },
            centralMoments: function (k, alpha, beta, loc) {
                const µ1 = this.nonCentralMoments(1, alpha, beta, loc);
                const µ2 = this.nonCentralMoments(2, alpha, beta, loc);
                const µ3 = this.nonCentralMoments(3, alpha, beta, loc);
                const µ4 = this.nonCentralMoments(4, alpha, beta, loc);

                let result;
                switch (k) {
                    case 1:
                        result = 0;
                        break;
                    case 2:
                        result = µ2 - µ1 ** 2;
                        break;
                    case 3:
                        result = µ3 - 3 * µ1 * µ2 + 2 * µ1 ** 3;
                        break;
                    case 4:
                        result = µ4 - 4 * µ1 * µ3 + 6 * µ1 ** 2 * µ2 - 3 * µ1 ** 4;
                        break;
                }
                return result;
            },
            stats: {
                mean: function (alpha, beta, loc) {
                    const µ1 = dists.gamma_3p.measurements.nonCentralMoments(1, alpha, beta, loc);
                    return loc + µ1;
                },
                variance: function (alpha, beta, loc) {
                    const µ1 = dists.gamma_3p.measurements.nonCentralMoments(1, alpha, beta, loc);
                    const µ2 = dists.gamma_3p.measurements.nonCentralMoments(2, alpha, beta, loc);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (alpha, beta, loc) {
                    return Math.sqrt(this.variance(alpha, beta, loc));
                },
                skewness: function (alpha, beta, loc) {
                    const central_µ3 = dists.gamma_3p.measurements.centralMoments(3, alpha, beta, loc);
                    return central_µ3 / this.standardDeviation(alpha, beta, loc) ** 3;
                },
                kurtosis: function (alpha, beta, loc) {
                    const central_µ4 = dists.gamma_3p.measurements.centralMoments(4, alpha, beta, loc);
                    return central_µ4 / this.standardDeviation(alpha, beta, loc) ** 4;
                },
                median: function (alpha, beta, loc) {
                    return dists.gamma_3p.measurements.ppf(0.5, alpha, beta, loc);
                },
                mode: function (alpha, beta, loc) {
                    return beta * (alpha - 1) + loc;
                },
            },
        },
    },
    generalized_extreme_value: {
        measurements: {
            nonCentralMoments: function (k, ξ, miu, sigma) {
                return jStat.gammafn(1 - ξ * k);
            },
            centralMoments: function (k, ξ, miu, sigma) {
                const µ1 = this.nonCentralMoments(1, ξ, miu, sigma);
                const µ2 = this.nonCentralMoments(2, ξ, miu, sigma);
                const µ3 = this.nonCentralMoments(3, ξ, miu, sigma);
                const µ4 = this.nonCentralMoments(4, ξ, miu, sigma);

                let result;
                switch (k) {
                    case 1:
                        result = 0;
                        break;
                    case 2:
                        result = µ2 - µ1 ** 2;
                        break;
                    case 3:
                        result = µ3 - 3 * µ1 * µ2 + 2 * µ1 ** 3;
                        break;
                    case 4:
                        result = µ4 - 4 * µ1 * µ3 + 6 * µ1 ** 2 * µ2 - 3 * µ1 ** 4;
                        break;
                }
                return result;
            },
            stats: {
                mean: function (ξ, miu, sigma) {
                    const µ1 = dists.generalized_extreme_value.measurements.nonCentralMoments(1, ξ, miu, sigma);
                    if (ξ == 0) {
                        return miu + sigma * 0.5772156649;
                    }
                    return miu + (sigma * (µ1 - 1)) / ξ;
                },
                variance: function (ξ, miu, sigma) {
                    const µ1 = dists.generalized_extreme_value.measurements.nonCentralMoments(1, ξ, miu, sigma);
                    const µ2 = dists.generalized_extreme_value.measurements.nonCentralMoments(2, ξ, miu, sigma);
                    if (ξ == 0) {
                        return sigma ** 2 * (Math.PI ** 2 / 6);
                    }
                    return (sigma ** 2 * (µ2 - µ1 ** 2)) / ξ ** 2;
                },
                standardDeviation: function (ξ, miu, sigma) {
                    return Math.sqrt(this.variance(ξ, miu, sigma));
                },
                skewness: function (ξ, miu, sigma) {
                    const central_µ3 = dists.generalized_extreme_value.measurements.centralMoments(3, ξ, miu, sigma);
                    const µ1 = dists.generalized_extreme_value.measurements.nonCentralMoments(1, ξ, miu, sigma);
                    const µ2 = dists.generalized_extreme_value.measurements.nonCentralMoments(2, ξ, miu, sigma);
                    const std = Math.sqrt(µ2 - µ1 ** 2);
                    if (ξ == 0) {
                        return (12 * Math.sqrt(6) * 1.20205690315959) / Math.PI ** 3;
                    }
                    return central_µ3 / std ** 3;
                },
                kurtosis: function (ξ, miu, sigma) {
                    const central_µ4 = dists.generalized_extreme_value.measurements.centralMoments(4, ξ, miu, sigma);
                    const µ1 = dists.generalized_extreme_value.measurements.nonCentralMoments(1, ξ, miu, sigma);
                    const µ2 = dists.generalized_extreme_value.measurements.nonCentralMoments(2, ξ, miu, sigma);
                    const std = Math.sqrt(µ2 - µ1 ** 2);
                    if (ξ == 0) {
                        return 5.4;
                    }
                    return central_µ4 / std ** 4;
                },
                median: function (ξ, miu, sigma) {
                    return dists.generalized_extreme_value.measurements.ppf(0.5, ξ, miu, sigma);
                },
                mode: function (ξ, miu, sigma) {
                    if (ξ == 0) {
                        return miu;
                    }
                    return miu + (sigma * ((1 + ξ) ** -ξ - 1)) / ξ;
                },
            },
        },
    },
    generalized_gamma: {
        measurements: {
            nonCentralMoments: function (k, a, d, p) {
                return (a ** k * jStat.gammafn((d + k) / p)) / jStat.gammafn(d / p);
            },
            centralMoments: function (k, a, d, p) {
                const µ1 = this.nonCentralMoments(1, a, d, p);
                const µ2 = this.nonCentralMoments(2, a, d, p);
                const µ3 = this.nonCentralMoments(3, a, d, p);
                const µ4 = this.nonCentralMoments(4, a, d, p);

                let result;
                switch (k) {
                    case 1:
                        result = 0;
                        break;
                    case 2:
                        result = µ2 - µ1 ** 2;
                        break;
                    case 3:
                        result = µ3 - 3 * µ1 * µ2 + 2 * µ1 ** 3;
                        break;
                    case 4:
                        result = µ4 - 4 * µ1 * µ3 + 6 * µ1 ** 2 * µ2 - 3 * µ1 ** 4;
                        break;
                }
                return result;
            },
            stats: {
                mean: function (a, d, p) {
                    const µ1 = dists.generalized_gamma.measurements.nonCentralMoments(1, a, d, p);
                    return µ1;
                },
                variance: function (a, d, p) {
                    const µ1 = dists.generalized_gamma.measurements.nonCentralMoments(1, a, d, p);
                    const µ2 = dists.generalized_gamma.measurements.nonCentralMoments(2, a, d, p);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (a, d, p) {
                    return Math.sqrt(this.variance(a, d, p));
                },
                skewness: function (a, d, p) {
                    const central_µ3 = dists.generalized_gamma.measurements.centralMoments(3, a, d, p);
                    return central_µ3 / this.standardDeviation(a, d, p) ** 3;
                },
                kurtosis: function (a, d, p) {
                    const central_µ4 = dists.generalized_gamma.measurements.centralMoments(4, a, d, p);
                    return central_µ4 / this.standardDeviation(a, d, p) ** 4;
                },
                median: function (a, d, p) {
                    return dists.generalized_gamma.measurements.ppf(0.5, a, d, p);
                },
                mode: function (a, d, p) {
                    return a * ((d - 1) / p) ** (1 / p);
                },
            },
        },
    },
    generalized_gamma_4p: {
        measurements: {
            nonCentralMoments: function (k, a, d, p, loc) {
                return (a ** k * jStat.gammafn((d + k) / p)) / jStat.gammafn(d / p);
            },
            centralMoments: function (k, a, d, p, loc) {
                const µ1 = this.nonCentralMoments(1, a, d, p, loc);
                const µ2 = this.nonCentralMoments(2, a, d, p, loc);
                const µ3 = this.nonCentralMoments(3, a, d, p, loc);
                const µ4 = this.nonCentralMoments(4, a, d, p, loc);

                let result;
                switch (k) {
                    case 1:
                        result = 0;
                        break;
                    case 2:
                        result = µ2 - µ1 ** 2;
                        break;
                    case 3:
                        result = µ3 - 3 * µ1 * µ2 + 2 * µ1 ** 3;
                        break;
                    case 4:
                        result = µ4 - 4 * µ1 * µ3 + 6 * µ1 ** 2 * µ2 - 3 * µ1 ** 4;
                        break;
                }
                return result;
            },
            stats: {
                mean: function (a, d, p, loc) {
                    const µ1 = dists.generalized_gamma_4p.measurements.nonCentralMoments(1, a, d, p, loc);
                    return loc + µ1;
                },
                variance: function (a, d, p, loc) {
                    const µ1 = dists.generalized_gamma_4p.measurements.nonCentralMoments(1, a, d, p, loc);
                    const µ2 = dists.generalized_gamma_4p.measurements.nonCentralMoments(2, a, d, p, loc);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (a, d, p, loc) {
                    return Math.sqrt(this.variance(a, d, p, loc));
                },
                skewness: function (a, d, p, loc) {
                    const central_µ3 = dists.generalized_gamma_4p.measurements.centralMoments(3, a, d, p, loc);
                    return central_µ3 / this.standardDeviation(a, d, p, loc) ** 3;
                },
                kurtosis: function (a, d, p, loc) {
                    const central_µ4 = dists.generalized_gamma_4p.measurements.centralMoments(4, a, d, p, loc);
                    return central_µ4 / this.standardDeviation(a, d, p, loc) ** 4;
                },
                median: function (a, d, p, loc) {
                    return dists.generalized_gamma_4p.measurements.ppf(0.5, a, d, p, loc);
                },
                mode: function (a, d, p, loc) {
                    return loc + a * ((d - 1) / p) ** (1 / p);
                },
            },
        },
    },
    generalized_logistic: {
        measurements: {
            nonCentralMoments: function (k, c, loc, scale) {
                return undefined;
            },
            centralMoments: function (k, c, loc, scale) {
                return undefined;
            },
            stats: {
                mean: function (c, loc, scale) {
                    return loc + scale * (0.57721 + digamma(c));
                },
                variance: function (c, loc, scale) {
                    return scale * scale * ((Math.PI * Math.PI) / 6 + polygamma(1, c));
                },
                standardDeviation: function (c, loc, scale) {
                    return Math.sqrt(this.variance(c, loc, scale));
                },
                skewness: function (c, loc, scale) {
                    return (2.40411380631918 + polygamma(2, c)) / ((Math.PI * Math.PI) / 6 + polygamma(1, c)) ** 1.5;
                },
                kurtosis: function (c, loc, scale) {
                    return 3 + (6.49393940226682 + polygamma(3, c)) / ((Math.PI * Math.PI) / 6 + polygamma(1, c)) ** 2;
                },
                median: function (c, loc, scale) {
                    return dists.generalized_logistic.measurements.ppf(0.5, c, loc, scale);
                },
                mode: function (c, loc, scale) {
                    return loc + scale * Math.log(c);
                },
            },
        },
    },
    generalized_normal: {
        measurements: {
            nonCentralMoments: function (k, miu, alpha, beta) {
                return undefined;
            },
            centralMoments: function (k, miu, alpha, beta) {
                return undefined;
            },
            stats: {
                mean: function (miu, alpha, beta) {
                    return beta;
                },
                variance: function (miu, alpha, beta) {
                    return (miu ** 2 * jStat.gammafn(3 / alpha)) / jStat.gammafn(1 / alpha);
                },
                standardDeviation: function (miu, alpha, beta) {
                    return Math.sqrt(this.variance(miu, alpha, beta));
                },
                skewness: function (miu, alpha, beta) {
                    return 0;
                },
                kurtosis: function (miu, alpha, beta) {
                    return (jStat.gammafn(5 / alpha) * jStat.gammafn(1 / alpha)) / jStat.gammafn(3 / alpha) ** 2;
                },
                median: function (miu, alpha, beta) {
                    return dists.generalized_normal.measurements.ppf(0.5, miu, alpha, beta);
                },
                mode: function (miu, alpha, beta) {
                    return beta;
                },
            },
        },
    },
    generalized_pareto: {
        measurements: {
            nonCentralMoments: function (k, c, miu, sigma) {
                return undefined;
            },
            centralMoments: function (k, c, miu, sigma) {
                return undefined;
            },
            stats: {
                mean: function (c, miu, sigma) {
                    return miu + sigma / (1 - c);
                },
                variance: function (c, miu, sigma) {
                    return (sigma * sigma) / ((1 - c) * (1 - c) * (1 - 2 * c));
                },
                standardDeviation: function (c, miu, sigma) {
                    return Math.sqrt(this.variance(c, miu, sigma));
                },
                skewness: function (c, miu, sigma) {
                    return (2 * (1 + c) * Math.sqrt(1 - 2 * c)) / (1 - 3 * c);
                },
                kurtosis: function (c, miu, sigma) {
                    return (3 * (1 - 2 * c) * (2 * c * c + c + 3)) / ((1 - 3 * c) * (1 - 4 * c));
                },
                median: function (c, miu, sigma) {
                    return dists.generalized_pareto.measurements.ppf(0.5, c, miu, sigma);
                },
                mode: function (c, miu, sigma) {
                    return miu;
                },
            },
        },
    },
    gibrat: {
        measurements: {
            nonCentralMoments: function (k, loc, scale) {
                return undefined;
            },
            centralMoments: function (k, loc, scale) {
                return undefined;
            },
            stats: {
                mean: function (loc, scale) {
                    return loc + scale * Math.sqrt(Math.exp(1));
                },
                variance: function (loc, scale) {
                    return Math.exp(1) * (Math.exp(1) - 1) * scale * scale;
                },
                standardDeviation: function (loc, scale) {
                    return Math.sqrt(this.variance(loc, scale));
                },
                skewness: function (loc, scale) {
                    return (2 + Math.exp(1)) * Math.sqrt(Math.exp(1) - 1);
                },
                kurtosis: function (loc, scale) {
                    return Math.exp(1) ** 4 + 2 * Math.exp(1) ** 3 + 3 * Math.exp(1) ** 2 - 6;
                },
                median: function (loc, scale) {
                    return dists.gibrat.measurements.ppf(0.5, loc, scale);
                },
                mode: function (loc, scale) {
                    return (1 / Math.exp(1)) * scale + loc;
                },
            },
        },
    },
    gumbel_left: {
        measurements: {
            nonCentralMoments: function (k, miu, sigma) {
                return undefined;
            },
            centralMoments: function (k, miu, sigma) {
                return undefined;
            },
            stats: {
                mean: function (miu, sigma) {
                    return miu - 0.5772156649 * sigma;
                },
                variance: function (miu, sigma) {
                    return sigma ** 2 * (Math.PI ** 2 / 6);
                },
                standardDeviation: function (miu, sigma) {
                    return Math.sqrt(this.variance(miu, sigma));
                },
                skewness: function (miu, sigma) {
                    return (-12 * Math.sqrt(6) * 1.20205690315959) / Math.PI ** 3;
                },
                kurtosis: function (miu, sigma) {
                    return 3 + 12 / 5;
                },
                median: function (miu, sigma) {
                    return dists.gumbel_left.measurements.ppf(0.5, miu, sigma);
                },
                mode: function (miu, sigma) {
                    return miu;
                },
            },
        },
    },
    gumbel_right: {
        measurements: {
            nonCentralMoments: function (k, miu, sigma) {
                return undefined;
            },
            centralMoments: function (k, miu, sigma) {
                return undefined;
            },
            stats: {
                mean: function (miu, sigma) {
                    return miu + 0.5772156649 * sigma;
                },
                variance: function (miu, sigma) {
                    return sigma ** 2 * (Math.PI ** 2 / 6);
                },
                standardDeviation: function (miu, sigma) {
                    return Math.sqrt(this.variance(miu, sigma));
                },
                skewness: function (miu, sigma) {
                    return (12 * Math.sqrt(6) * 1.20205690315959) / Math.PI ** 3;
                },
                kurtosis: function (miu, sigma) {
                    return 3 + 12 / 5;
                },
                median: function (miu, sigma) {
                    return dists.gumbel_right.measurements.ppf(0.5, miu, sigma);
                },
                mode: function (miu, sigma) {
                    return miu;
                },
            },
        },
    },
    half_normal: {
        measurements: {
            nonCentralMoments: function (k, miu, sigma) {
                return undefined;
            },
            centralMoments: function (k, miu, sigma) {
                return undefined;
            },
            stats: {
                mean: function (miu, sigma) {
                    return miu + sigma * Math.sqrt(2 / Math.PI);
                },
                variance: function (miu, sigma) {
                    return sigma * sigma * (1 - 2 / Math.PI);
                },
                standardDeviation: function (miu, sigma) {
                    return Math.sqrt(this.variance(miu, sigma));
                },
                skewness: function (miu, sigma) {
                    return (Math.sqrt(2) * (4 - Math.PI)) / (Math.PI - 2) ** 1.5;
                },
                kurtosis: function (miu, sigma) {
                    return 3 + (8 * (Math.PI - 3)) / (Math.PI - 2) ** 2;
                },
                median: function (miu, sigma) {
                    return dists.half_normal.measurements.ppf(0.5, miu, sigma);
                },
                mode: function (miu, sigma) {
                    return miu;
                },
            },
        },
    },
    hyperbolic_secant: {
        measurements: {
            nonCentralMoments: function (k, miu, sigma) {
                return undefined;
            },
            centralMoments: function (k, miu, sigma) {
                return undefined;
            },
            stats: {
                mean: function (miu, sigma) {
                    return miu;
                },
                variance: function (miu, sigma) {
                    return sigma ** 2;
                },
                standardDeviation: function (miu, sigma) {
                    return Math.sqrt(this.variance(miu, sigma));
                },
                skewness: function (miu, sigma) {
                    return 0;
                },
                kurtosis: function (miu, sigma) {
                    return 5;
                },
                median: function (miu, sigma) {
                    return dists.hyperbolic_secant.measurements.ppf(0.5, miu, sigma);
                },
                mode: function (miu, sigma) {
                    return miu;
                },
            },
        },
    },
    inverse_gamma: {
        measurements: {
            nonCentralMoments: function (k, alpha, beta) {
                let result;
                switch (k) {
                    case 1:
                        result = beta ** k / (alpha - 1);
                        break;
                    case 2:
                        result = beta ** k / ((alpha - 1) * (alpha - 2));
                        break;
                    case 3:
                        result = beta ** k / ((alpha - 1) * (alpha - 2) * (alpha - 3));
                        break;
                    case 4:
                        result = beta ** k / ((alpha - 1) * (alpha - 2) * (alpha - 3) * (alpha - 4));
                        break;
                }
                return result;
            },
            centralMoments: function (k, alpha, beta) {
                const µ1 = this.nonCentralMoments(1, alpha, beta);
                const µ2 = this.nonCentralMoments(2, alpha, beta);
                const µ3 = this.nonCentralMoments(3, alpha, beta);
                const µ4 = this.nonCentralMoments(4, alpha, beta);

                let result;
                switch (k) {
                    case 1:
                        result = 0;
                        break;
                    case 2:
                        result = µ2 - µ1 ** 2;
                        break;
                    case 3:
                        result = µ3 - 3 * µ1 * µ2 + 2 * µ1 ** 3;
                        break;
                    case 4:
                        result = µ4 - 4 * µ1 * µ3 + 6 * µ1 ** 2 * µ2 - 3 * µ1 ** 4;
                        break;
                }
                return result;
            },
            stats: {
                mean: function (alpha, beta) {
                    const µ1 = dists.inverse_gamma.measurements.nonCentralMoments(1, alpha, beta);
                    return µ1;
                },
                variance: function (alpha, beta) {
                    const µ1 = dists.inverse_gamma.measurements.nonCentralMoments(1, alpha, beta);
                    const µ2 = dists.inverse_gamma.measurements.nonCentralMoments(2, alpha, beta);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (alpha, beta) {
                    return Math.sqrt(this.variance(alpha, beta));
                },
                skewness: function (alpha, beta) {
                    const central_µ3 = dists.inverse_gamma.measurements.centralMoments(3, alpha, beta);
                    return central_µ3 / this.standardDeviation(alpha, beta) ** 3;
                },
                kurtosis: function (alpha, beta) {
                    const central_µ4 = dists.inverse_gamma.measurements.centralMoments(4, alpha, beta);
                    return central_µ4 / this.standardDeviation(alpha, beta) ** 4;
                },
                median: function (alpha, beta) {
                    return dists.inverse_gamma.measurements.ppf(0.5, alpha, beta);
                },
                mode: function (alpha, beta) {
                    return beta / (alpha + 1);
                },
            },
        },
    },
    inverse_gamma_3p: {
        measurements: {
            nonCentralMoments: function (k, alpha, beta, loc) {
                let result;
                switch (k) {
                    case 1:
                        result = beta ** k / (alpha - 1);
                        break;
                    case 2:
                        result = beta ** k / ((alpha - 1) * (alpha - 2));
                        break;
                    case 3:
                        result = beta ** k / ((alpha - 1) * (alpha - 2) * (alpha - 3));
                        break;
                    case 4:
                        result = beta ** k / ((alpha - 1) * (alpha - 2) * (alpha - 3) * (alpha - 4));
                        break;
                }
                return result;
            },
            centralMoments: function (k, alpha, beta, loc) {
                const µ1 = this.nonCentralMoments(1, alpha, beta, loc);
                const µ2 = this.nonCentralMoments(2, alpha, beta, loc);
                const µ3 = this.nonCentralMoments(3, alpha, beta, loc);
                const µ4 = this.nonCentralMoments(4, alpha, beta, loc);

                let result;
                switch (k) {
                    case 1:
                        result = 0;
                        break;
                    case 2:
                        result = µ2 - µ1 ** 2;
                        break;
                    case 3:
                        result = µ3 - 3 * µ1 * µ2 + 2 * µ1 ** 3;
                        break;
                    case 4:
                        result = µ4 - 4 * µ1 * µ3 + 6 * µ1 ** 2 * µ2 - 3 * µ1 ** 4;
                        break;
                }
                return result;
            },
            stats: {
                mean: function (alpha, beta, loc) {
                    const µ1 = dists.inverse_gamma_3p.measurements.nonCentralMoments(1, alpha, beta, loc);
                    return loc + µ1;
                },
                variance: function (alpha, beta, loc) {
                    const µ1 = dists.inverse_gamma_3p.measurements.nonCentralMoments(1, alpha, beta, loc);
                    const µ2 = dists.inverse_gamma_3p.measurements.nonCentralMoments(2, alpha, beta, loc);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (alpha, beta, loc) {
                    return Math.sqrt(this.variance(alpha, beta, loc));
                },
                skewness: function (alpha, beta, loc) {
                    const central_µ3 = dists.inverse_gamma_3p.measurements.centralMoments(3, alpha, beta, loc);
                    return central_µ3 / this.standardDeviation(alpha, beta, loc) ** 3;
                },
                kurtosis: function (alpha, beta, loc) {
                    const central_µ4 = dists.inverse_gamma_3p.measurements.centralMoments(4, alpha, beta, loc);
                    return central_µ4 / this.standardDeviation(alpha, beta, loc) ** 4;
                },
                median: function (alpha, beta, loc) {
                    return dists.inverse_gamma_3p.measurements.ppf(0.5, alpha, beta, loc);
                },
                mode: function (alpha, beta, loc) {
                    return beta / (alpha + 1) + loc;
                },
            },
        },
    },
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
                    return miu ** 3 / lambda;
                },
                standardDeviation: function (miu, lambda) {
                    return Math.sqrt(this.variance(miu, lambda));
                },
                skewness: function (miu, lambda) {
                    return 3 * Math.sqrt(miu / lambda);
                },
                kurtosis: function (miu, lambda) {
                    return 15 * (miu / lambda) + 3;
                },
                median: function (miu, lambda) {
                    return dists.inverse_gaussian.measurements.ppf(0.5, miu, lambda);
                },
                mode: function (miu, lambda) {
                    return miu * (Math.sqrt(1 + (9 * miu * miu) / (4 * lambda * lambda)) - (3 * miu) / (2 * lambda));
                },
            },
        },
    },
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
                    return miu + loc;
                },
                variance: function (miu, lambda, loc) {
                    return miu ** 3 / lambda;
                },
                standardDeviation: function (miu, lambda, loc) {
                    return Math.sqrt(this.variance(miu, lambda, loc));
                },
                skewness: function (miu, lambda, loc) {
                    return 3 * Math.sqrt(miu / lambda);
                },
                kurtosis: function (miu, lambda, loc) {
                    return 15 * (miu / lambda) + 3;
                },
                median: function (miu, lambda, loc) {
                    return dists.inverse_gaussian_3p.measurements.ppf(0.5, miu, lambda, loc);
                },
                mode: function (miu, lambda, loc) {
                    return loc + miu * (Math.sqrt(1 + (9 * miu * miu) / (4 * lambda * lambda)) - (3 * miu) / (2 * lambda));
                },
            },
        },
    },
    johnson_sb: {
        measurements: {
            nonCentralMoments: function (k, xi, lambda, gamma, delta) {
                return undefined;
            },
            centralMoments: function (k, xi, lambda, gamma, delta) {
                return undefined;
            },
            stats: {
                mean: function (xi, lambda, gamma, delta) {
                    return undefined;
                },
                variance: function (xi, lambda, gamma, delta) {
                    return undefined;
                },
                standardDeviation: function (xi, lambda, gamma, delta) {
                    return Math.sqrt(this.variance(xi, lambda, gamma, delta));
                },
                skewness: function (xi, lambda, gamma, delta) {
                    return undefined;
                },
                kurtosis: function (xi, lambda, gamma, delta) {
                    return undefined;
                },
                median: function (xi, lambda, gamma, delta) {
                    return dists.johnson_sb.measurements.ppf(0.5, xi, lambda, gamma, delta);
                },
                mode: function (xi, lambda, gamma, delta) {
                    return undefined;
                },
            },
        },
    },
    johnson_su: {
        measurements: {
            nonCentralMoments: function (k, xi, lambda, gamma, delta) {
                return undefined;
            },
            centralMoments: function (k, xi, lambda, gamma, delta) {
                return undefined;
            },
            stats: {
                mean: function (xi, lambda, gamma, delta) {
                    return xi - lambda * Math.exp(delta ** -2 / 2) * Math.sinh(gamma / delta);
                },
                variance: function (xi, lambda, gamma, delta) {
                    return (lambda ** 2 / 2) * (Math.exp(delta ** -2) - 1) * (Math.exp(delta ** -2) * Math.cosh((2 * gamma) / delta) + 1);
                },
                standardDeviation: function (xi, lambda, gamma, delta) {
                    return Math.sqrt(this.variance(xi, lambda, gamma, delta));
                },
                skewness: function (xi, lambda, gamma, delta) {
                    return (
                        -(
                            lambda ** 3 *
                            Math.sqrt(Math.exp(delta ** -2)) *
                            (Math.exp(delta ** -2) - 1) ** 2 *
                            (Math.exp(delta ** -2) * (Math.exp(delta ** -2) + 2) * Math.sinh(3 * (gamma / delta)) + 3 * Math.sinh(gamma / delta))
                        ) /
                        (4 * this.standardDeviation(xi, lambda, gamma, delta) ** 3)
                    );
                },
                kurtosis: function (xi, lambda, gamma, delta) {
                    return (
                        (lambda ** 4 *
                            (Math.exp(delta ** -2) - 1) ** 2 *
                            (Math.exp(delta ** -2) ** 2 *
                                (Math.exp(delta ** -2) ** 4 + 2 * Math.exp(delta ** -2) ** 3 + 3 * Math.exp(delta ** -2) ** 2 - 3) *
                                Math.cosh(4 * (gamma / delta)) +
                                4 * Math.exp(delta ** -2) ** 2 * (Math.exp(delta ** -2) + 2) * Math.cosh(2 * (gamma / delta)) +
                                3 * (2 * Math.exp(delta ** -2) + 1))) /
                        (8 * this.standardDeviation(xi, lambda, gamma, delta) ** 4)
                    );
                },
                median: function (xi, lambda, gamma, delta) {
                    return dists.johnson_su.measurements.ppf(0.5, xi, lambda, gamma, delta);
                },
                mode: function (xi, lambda, gamma, delta) {
                    return undefined;
                },
            },
        },
    },
    kumaraswamy: {
        measurements: {
            nonCentralMoments: function (k, alpha, beta, min, max) {
                return (beta * jStat.gammafn(1 + k / alpha) * jStat.gammafn(beta)) / jStat.gammafn(1 + beta + k / alpha);
            },
            centralMoments: function (k, alpha, beta, min, max) {
                const µ1 = this.nonCentralMoments(1, alpha, beta, min, max);
                const µ2 = this.nonCentralMoments(2, alpha, beta, min, max);
                const µ3 = this.nonCentralMoments(3, alpha, beta, min, max);
                const µ4 = this.nonCentralMoments(4, alpha, beta, min, max);

                let result;
                switch (k) {
                    case 1:
                        result = 0;
                        break;
                    case 2:
                        result = µ2 - µ1 ** 2;
                        break;
                    case 3:
                        result = µ3 - 3 * µ1 * µ2 + 2 * µ1 ** 3;
                        break;
                    case 4:
                        result = µ4 - 4 * µ1 * µ3 + 6 * µ1 ** 2 * µ2 - 3 * µ1 ** 4;
                        break;
                }
                return result;
            },
            stats: {
                mean: function (alpha, beta, min, max) {
                    const µ1 = dists.kumaraswamy.measurements.nonCentralMoments(1, alpha, beta, min, max);
                    return min + (max - min) * µ1;
                },
                variance: function (alpha, beta, min, max) {
                    const µ1 = dists.kumaraswamy.measurements.nonCentralMoments(1, alpha, beta, min, max);
                    const µ2 = dists.kumaraswamy.measurements.nonCentralMoments(2, alpha, beta, min, max);
                    return (max - min) ** 2 * (µ2 - µ1 ** 2);
                },
                standardDeviation: function (alpha, beta, min, max) {
                    return Math.sqrt(this.variance(alpha, beta, min, max));
                },
                skewness: function (alpha, beta, min, max) {
                    const central_µ3 = dists.kumaraswamy.measurements.centralMoments(3, alpha, beta, min, max);
                    const µ1 = dists.kumaraswamy.measurements.nonCentralMoments(1, alpha, beta, min, max);
                    const µ2 = dists.kumaraswamy.measurements.nonCentralMoments(2, alpha, beta, min, max);
                    const std = Math.sqrt(µ2 - µ1 ** 2);
                    return central_µ3 / std ** 3;
                },
                kurtosis: function (alpha, beta, min, max) {
                    const central_µ4 = dists.kumaraswamy.measurements.centralMoments(4, alpha, beta, min, max);
                    const µ1 = dists.kumaraswamy.measurements.nonCentralMoments(1, alpha, beta, min, max);
                    const µ2 = dists.kumaraswamy.measurements.nonCentralMoments(2, alpha, beta, min, max);
                    const std = Math.sqrt(µ2 - µ1 ** 2);
                    return central_µ4 / std ** 4;
                },
                median: function (alpha, beta, min, max) {
                    return dists.kumaraswamy.measurements.ppf(0.5, alpha, beta, min, max);
                },
                mode: function (alpha, beta, min, max) {
                    return ((alpha - 1) / (alpha * beta - 1)) ** (1 / alpha) * min + (max - min);
                },
            },
        },
    },
    laplace: {
        measurements: {
            nonCentralMoments: function (k, miu, b) {
                return undefined;
            },
            centralMoments: function (k, miu, b) {
                return undefined;
            },
            stats: {
                mean: function (miu, b) {
                    return miu;
                },
                variance: function (miu, b) {
                    return 2 * b ** 2;
                },
                standardDeviation: function (miu, b) {
                    return Math.sqrt(this.variance(miu, b));
                },
                skewness: function (miu, b) {
                    return 0;
                },
                kurtosis: function (miu, b) {
                    return 6;
                },
                median: function (miu, b) {
                    return dists.laplace.measurements.ppf(0.5, miu, b);
                },
                mode: function (miu, b) {
                    return miu;
                },
            },
        },
    },
    levy: {
        measurements: {
            nonCentralMoments: function (k, miu, c) {
                return undefined;
            },
            centralMoments: function (k, miu, c) {
                return undefined;
            },
            stats: {
                mean: function (miu, c) {
                    return Infinity;
                },
                variance: function (miu, c) {
                    return Infinity;
                },
                standardDeviation: function (miu, c) {
                    return Math.sqrt(this.variance(miu, c));
                },
                skewness: function (miu, c) {
                    return undefined;
                },
                kurtosis: function (miu, c) {
                    return undefined;
                },
                median: function (miu, c) {
                    return dists.levy.measurements.ppf(0.5, miu, c);
                },
                mode: function (miu, c) {
                    return miu + c / 3;
                },
            },
        },
    },
    loggamma: {
        measurements: {
            nonCentralMoments: function (k, c, miu, sigma) {
                return undefined;
            },
            centralMoments: function (k, c, miu, sigma) {
                return undefined;
            },
            stats: {
                mean: function (c, miu, sigma) {
                    return digamma(c) * sigma + miu;
                },
                variance: function (c, miu, sigma) {
                    return polygamma(1, c) * sigma * sigma;
                },
                standardDeviation: function (c, miu, sigma) {
                    return Math.sqrt(this.variance(c, miu, sigma));
                },
                skewness: function (c, miu, sigma) {
                    return polygamma(2, c) / polygamma(1, c) ** 1.5;
                },
                kurtosis: function (c, miu, sigma) {
                    return polygamma(3, c) / polygamma(1, c) ** 2 + 3;
                },
                median: function (c, miu, sigma) {
                    return dists.loggamma.measurements.ppf(0.5, c, miu, sigma);
                },
                mode: function (c, miu, sigma) {
                    return miu + sigma * Math.log(c);
                },
            },
        },
    },
    logistic: {
        measurements: {
            nonCentralMoments: function (k, miu, sigma) {
                return undefined;
            },
            centralMoments: function (k, miu, sigma) {
                return undefined;
            },
            stats: {
                mean: function (miu, sigma) {
                    return miu;
                },
                variance: function (miu, sigma) {
                    return (sigma * sigma * Math.PI * Math.PI) / 3;
                },
                standardDeviation: function (miu, sigma) {
                    return Math.sqrt(this.variance(miu, sigma));
                },
                skewness: function (miu, sigma) {
                    return 0;
                },
                kurtosis: function (miu, sigma) {
                    return 4.2;
                },
                median: function (miu, sigma) {
                    return dists.logistic.measurements.ppf(0.5, miu, sigma);
                },
                mode: function (miu, sigma) {
                    return miu;
                },
            },
        },
    },
    loglogistic: {
        measurements: {
            nonCentralMoments: function (k, alpha, beta) {
                return (alpha ** k * ((k * Math.PI) / beta)) / Math.sin((k * Math.PI) / beta);
            },
            centralMoments: function (k, alpha, beta) {
                const µ1 = this.nonCentralMoments(1, alpha, beta);
                const µ2 = this.nonCentralMoments(2, alpha, beta);
                const µ3 = this.nonCentralMoments(3, alpha, beta);
                const µ4 = this.nonCentralMoments(4, alpha, beta);

                let result;
                switch (k) {
                    case 1:
                        result = 0;
                        break;
                    case 2:
                        result = µ2 - µ1 ** 2;
                        break;
                    case 3:
                        result = µ3 - 3 * µ1 * µ2 + 2 * µ1 ** 3;
                        break;
                    case 4:
                        result = µ4 - 4 * µ1 * µ3 + 6 * µ1 ** 2 * µ2 - 3 * µ1 ** 4;
                        break;
                }
                return result;
            },
            stats: {
                mean: function (alpha, beta) {
                    const µ1 = dists.loglogistic.measurements.nonCentralMoments(1, alpha, beta);
                    return µ1;
                },
                variance: function (alpha, beta) {
                    const µ1 = dists.loglogistic.measurements.nonCentralMoments(1, alpha, beta);
                    const µ2 = dists.loglogistic.measurements.nonCentralMoments(2, alpha, beta);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (alpha, beta) {
                    return Math.sqrt(this.variance(alpha, beta));
                },
                skewness: function (alpha, beta) {
                    const central_µ3 = dists.loglogistic.measurements.centralMoments(3, alpha, beta);
                    return central_µ3 / this.standardDeviation(alpha, beta) ** 3;
                },
                kurtosis: function (alpha, beta) {
                    const central_µ4 = dists.loglogistic.measurements.centralMoments(4, alpha, beta);
                    return central_µ4 / this.standardDeviation(alpha, beta) ** 4;
                },
                median: function (alpha, beta) {
                    return dists.loglogistic.measurements.ppf(0.5, alpha, beta);
                },
                mode: function (alpha, beta) {
                    return alpha * ((beta - 1) / (beta + 1)) ** (1 / beta);
                },
            },
        },
    },
    loglogistic_3p: {
        measurements: {
            nonCentralMoments: function (k, alpha, beta, loc) {
                return (alpha ** k * ((k * Math.PI) / beta)) / Math.sin((k * Math.PI) / beta);
            },
            centralMoments: function (k, alpha, beta, loc) {
                const µ1 = this.nonCentralMoments(1, alpha, beta, loc);
                const µ2 = this.nonCentralMoments(2, alpha, beta, loc);
                const µ3 = this.nonCentralMoments(3, alpha, beta, loc);
                const µ4 = this.nonCentralMoments(4, alpha, beta, loc);

                let result;
                switch (k) {
                    case 1:
                        result = 0;
                        break;
                    case 2:
                        result = µ2 - µ1 ** 2;
                        break;
                    case 3:
                        result = µ3 - 3 * µ1 * µ2 + 2 * µ1 ** 3;
                        break;
                    case 4:
                        result = µ4 - 4 * µ1 * µ3 + 6 * µ1 ** 2 * µ2 - 3 * µ1 ** 4;
                        break;
                }
                return result;
            },
            stats: {
                mean: function (alpha, beta, loc) {
                    const µ1 = dists.loglogistic_3p.measurements.nonCentralMoments(1, alpha, beta, loc);
                    return loc + µ1;
                },
                variance: function (alpha, beta, loc) {
                    const µ1 = dists.loglogistic_3p.measurements.nonCentralMoments(1, alpha, beta, loc);
                    const µ2 = dists.loglogistic_3p.measurements.nonCentralMoments(2, alpha, beta, loc);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (alpha, beta, loc) {
                    return Math.sqrt(this.variance(alpha, beta, loc));
                },
                skewness: function (alpha, beta, loc) {
                    const central_µ3 = dists.loglogistic_3p.measurements.centralMoments(3, alpha, beta, loc);
                    return central_µ3 / this.standardDeviation(alpha, beta, loc) ** 3;
                },
                kurtosis: function (alpha, beta, loc) {
                    const central_µ4 = dists.loglogistic_3p.measurements.centralMoments(4, alpha, beta, loc);
                    return central_µ4 / this.standardDeviation(alpha, beta, loc) ** 4;
                },
                median: function (alpha, beta, loc) {
                    return dists.loglogistic_3p.measurements.ppf(0.5, alpha, beta, loc);
                },
                mode: function (alpha, beta, loc) {
                    return loc + alpha * ((beta - 1) / (beta + 1)) ** (1 / beta);
                },
            },
        },
    },
    lognormal: {
        measurements: {
            nonCentralMoments: function (k, miu, sigma) {
                return undefined;
            },
            centralMoments: function (k, miu, sigma) {
                return undefined;
            },
            stats: {
                mean: function (miu, sigma) {
                    return Math.exp(miu + sigma ** 2 / 2);
                },
                variance: function (miu, sigma) {
                    return (Math.exp(sigma ** 2) - 1) * Math.exp(2 * miu + sigma ** 2);
                },
                standardDeviation: function (miu, sigma) {
                    return Math.sqrt(this.variance(miu, sigma));
                },
                skewness: function (miu, sigma) {
                    return (Math.exp(sigma * sigma) + 2) * Math.sqrt(Math.exp(sigma * sigma) - 1);
                },
                kurtosis: function (miu, sigma) {
                    return Math.exp(4 * sigma * sigma) + 2 * Math.exp(3 * sigma * sigma) + 3 * Math.exp(2 * sigma * sigma) - 3;
                },
                median: function (miu, sigma) {
                    return dists.lognormal.measurements.ppf(0.5, miu, sigma);
                },
                mode: function (miu, sigma) {
                    return Math.exp(miu - sigma * sigma);
                },
            },
        },
    },
    maxwell: {
        measurements: {
            nonCentralMoments: function (k, alpha, loc) {
                return undefined;
            },
            centralMoments: function (k, alpha, loc) {
                return undefined;
            },
            stats: {
                mean: function (alpha, loc) {
                    return 2 * Math.sqrt(2 / Math.PI) * alpha + loc;
                },
                variance: function (alpha, loc) {
                    return (alpha * alpha * (3 * Math.PI - 8)) / Math.PI;
                },
                standardDeviation: function (alpha, loc) {
                    return Math.sqrt(this.variance(alpha, loc));
                },
                skewness: function (alpha, loc) {
                    return (2 * Math.sqrt(2) * (16 - 5 * Math.PI)) / (3 * Math.PI - 8) ** 1.5;
                },
                kurtosis: function (alpha, loc) {
                    return (4 * (-96 + 40 * Math.PI - 3 * Math.PI * Math.PI)) / (3 * Math.PI - 8) ** 2 + 3;
                },
                median: function (alpha, loc) {
                    return dists.maxwell.measurements.ppf(0.5, alpha, loc);
                },
                mode: function (alpha, loc) {
                    return Math.sqrt(2) * alpha + loc;
                },
            },
        },
    },
    moyal: {
        measurements: {
            nonCentralMoments: function (k, miu, sigma) {
                return undefined;
            },
            centralMoments: function (k, miu, sigma) {
                return undefined;
            },
            stats: {
                mean: function (miu, sigma) {
                    return miu + sigma * (Math.log(2) + 0.577215664901532);
                },
                variance: function (miu, sigma) {
                    return (sigma * sigma * Math.PI * Math.PI) / 2;
                },
                standardDeviation: function (miu, sigma) {
                    return Math.sqrt(this.variance(miu, sigma));
                },
                skewness: function (miu, sigma) {
                    return 1.5351415907229;
                },
                kurtosis: function (miu, sigma) {
                    return 7;
                },
                median: function (miu, sigma) {
                    return dists.moyal.measurements.ppf(0.5, miu, sigma);
                },
                mode: function (miu, sigma) {
                    return miu;
                },
            },
        },
    },
    nakagami: {
        measurements: {
            nonCentralMoments: function (k, m, omega) {
                return undefined;
            },
            centralMoments: function (k, m, omega) {
                return undefined;
            },
            stats: {
                mean: function (m, omega) {
                    return (jStat.gammafn(m + 0.5) / jStat.gammafn(m)) * Math.sqrt(omega / m);
                },
                variance: function (m, omega) {
                    return omega * (1 - (1 / m) * (jStat.gammafn(m + 0.5) / jStat.gammafn(m)) ** 2);
                },
                standardDeviation: function (m, omega) {
                    return Math.sqrt(this.variance(m, omega));
                },
                skewness: function (m, omega) {
                    return (
                        ((jStat.gammafn(m + 0.5) / jStat.gammafn(m)) *
                            Math.sqrt(1 / m) *
                            (1 - 4 * m * (1 - ((jStat.gammafn(m + 0.5) / jStat.gammafn(m)) * Math.sqrt(1 / m)) ** 2))) /
                        (2 * m * (1 - ((jStat.gammafn(m + 0.5) / jStat.gammafn(m)) * Math.sqrt(1 / m)) ** 2) ** 1.5)
                    );
                },
                kurtosis: function (m, omega) {
                    return (
                        3 +
                        (-6 * ((jStat.gammafn(m + 0.5) / jStat.gammafn(m)) * Math.sqrt(1 / m)) ** 4 * m +
                            (8 * m - 2) * ((jStat.gammafn(m + 0.5) / jStat.gammafn(m)) * Math.sqrt(1 / m)) ** 2 -
                            2 * m +
                            1) /
                            (m * (1 - ((jStat.gammafn(m + 0.5) / jStat.gammafn(m)) * Math.sqrt(1 / m)) ** 2) ** 2)
                    );
                },
                median: function (m, omega) {
                    return dists.nakagami.measurements.ppf(0.5, m, omega);
                },
                mode: function (m, omega) {
                    return (Math.sqrt(2) / 2) * Math.sqrt((omega * (2 * m - 1)) / m);
                },
            },
        },
    },
    nc_chi_square: {
        measurements: {
            nonCentralMoments: function (k, lambda, n) {
                return undefined;
            },
            centralMoments: function (k, lambda, n) {
                return undefined;
            },
            stats: {
                mean: function (lambda, n) {
                    return lambda + n;
                },
                variance: function (lambda, n) {
                    return 2 * (n + 2 * lambda);
                },
                standardDeviation: function (lambda, n) {
                    return Math.sqrt(this.variance(lambda, n));
                },
                skewness: function (lambda, n) {
                    return (2 ** 1.5 * (n + 3 * lambda)) / (n + 2 * lambda) ** 1.5;
                },
                kurtosis: function (lambda, n) {
                    return 3 + (12 * (n + 4 * lambda)) / (n + 2 * lambda) ** 2;
                },
                median: function (lambda, n) {
                    return dists.nc_chi_square.measurements.ppf(0.5, lambda, n);
                },
                mode: function (lambda, n) {
                    return undefined;
                },
            },
        },
    },
    nc_f: {
        measurements: {
            nonCentralMoments: function (k, lambda, n1, n2) {
                let result;
                switch (k) {
                    case 1:
                        result = (n2 / n1) * ((n1 + lambda) / (n2 - 2));
                        break;
                    case 2:
                        result = (n2 / n1) ** 2 * (1 / ((n2 - 2) * (n2 - 4))) * (lambda ** 2 + (2 * lambda + n1) * (n1 + 2));
                        break;
                    case 3:
                        result = (n2 / n1) ** 3 * (1 / ((n2 - 2) * (n2 - 4) * (n2 - 6))) * (lambda ** 3 + 3 * (n1 + 4) * lambda ** 2 + (3 * lambda + n1) * (n1 + 4) * (n1 + 2));
                        break;
                    case 4:
                        result =
                            (n2 / n1) ** 4 *
                            (1 / ((n2 - 2) * (n2 - 4) * (n2 - 6) * (n2 - 8))) *
                            (lambda ** 4 + 4 * (n1 + 6) * lambda ** 3 + 6 * (n1 + 6) * (n1 + 4) * lambda ** 2 + (4 * lambda + n1) * (n1 + 2) * (n1 + 4) * (n1 + 6));
                        break;
                }
                return result;
            },
            centralMoments: function (k, lambda, n1, n2) {
                const µ1 = this.nonCentralMoments(1, lambda, n1, n2);
                const µ2 = this.nonCentralMoments(2, lambda, n1, n2);
                const µ3 = this.nonCentralMoments(3, lambda, n1, n2);
                const µ4 = this.nonCentralMoments(4, lambda, n1, n2);

                let result;
                switch (k) {
                    case 1:
                        result = 0;
                        break;
                    case 2:
                        result = µ2 - µ1 ** 2;
                        break;
                    case 3:
                        result = µ3 - 3 * µ1 * µ2 + 2 * µ1 ** 3;
                        break;
                    case 4:
                        result = µ4 - 4 * µ1 * µ3 + 6 * µ1 ** 2 * µ2 - 3 * µ1 ** 4;
                        break;
                }
                return result;
            },
            stats: {
                mean: function (lambda, n1, n2) {
                    const µ1 = dists.nc_f.measurements.nonCentralMoments(1, lambda, n1, n2);
                    return µ1;
                },
                variance: function (lambda, n1, n2) {
                    const µ1 = dists.nc_f.measurements.nonCentralMoments(1, lambda, n1, n2);
                    const µ2 = dists.nc_f.measurements.nonCentralMoments(2, lambda, n1, n2);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (lambda, n1, n2) {
                    return Math.sqrt(this.variance(lambda, n1, n2));
                },
                skewness: function (lambda, n1, n2) {
                    const central_µ3 = dists.nc_f.measurements.centralMoments(3, lambda, n1, n2);
                    return central_µ3 / this.standardDeviation(lambda, n1, n2) ** 3;
                },
                kurtosis: function (lambda, n1, n2) {
                    const central_µ4 = dists.nc_f.measurements.centralMoments(4, lambda, n1, n2);
                    return central_µ4 / this.standardDeviation(lambda, n1, n2) ** 4;
                },
                median: function (lambda, n1, n2) {
                    return dists.nc_f.measurements.ppf(0.5, lambda, n1, n2);
                },
                mode: function (lambda, n1, n2) {
                    return undefined;
                },
            },
        },
    },
    nc_t_student: {
        measurements: {
            nonCentralMoments: function (k, lambda, n, loc, scale) {
                let result;
                switch (k) {
                    case 1:
                        result = (lambda * Math.sqrt(n / 2) * jStat.gammafn((n - 1) / 2)) / jStat.gammafn(n / 2);
                        break;
                    case 2:
                        result = (n * (1 + lambda * lambda)) / (n - 2);
                        break;
                    case 3:
                        result = (n ** 1.5 * Math.sqrt(2) * jStat.gammafn((n - 3) / 2) * lambda * (3 + lambda * lambda)) / (4 * jStat.gammafn(n / 2));
                        break;
                    case 4:
                        result = (n * n * (lambda ** 4 + 6 * lambda ** 2 + 3)) / ((n - 2) * (n - 4));
                        break;
                }
                return result;
            },
            centralMoments: function (k, lambda, n, loc, scale) {
                const µ1 = this.nonCentralMoments(1, lambda, n, loc, scale);
                const µ2 = this.nonCentralMoments(2, lambda, n, loc, scale);
                const µ3 = this.nonCentralMoments(3, lambda, n, loc, scale);
                const µ4 = this.nonCentralMoments(4, lambda, n, loc, scale);

                let result;
                switch (k) {
                    case 1:
                        result = 0;
                        break;
                    case 2:
                        result = µ2 - µ1 ** 2;
                        break;
                    case 3:
                        result = µ3 - 3 * µ1 * µ2 + 2 * µ1 ** 3;
                        break;
                    case 4:
                        result = µ4 - 4 * µ1 * µ3 + 6 * µ1 ** 2 * µ2 - 3 * µ1 ** 4;
                        break;
                }
                return result;
            },
            stats: {
                mean: function (lambda, n, loc, scale) {
                    const µ1 = dists.nc_t_student.measurements.nonCentralMoments(1, lambda, n, loc, scale);
                    return loc + scale * µ1;
                },
                variance: function (lambda, n, loc, scale) {
                    const µ1 = dists.nc_t_student.measurements.nonCentralMoments(1, lambda, n, loc, scale);
                    const µ2 = dists.nc_t_student.measurements.nonCentralMoments(2, lambda, n, loc, scale);
                    return scale ** 2 * (µ2 - µ1 ** 2);
                },
                standardDeviation: function (lambda, n, loc, scale) {
                    return Math.sqrt(this.variance(lambda, n, loc, scale));
                },
                skewness: function (lambda, n, loc, scale) {
                    const µ1 = dists.nc_t_student.measurements.nonCentralMoments(1, lambda, n, loc, scale);
                    const µ2 = dists.nc_t_student.measurements.nonCentralMoments(2, lambda, n, loc, scale);
                    const central_µ3 = dists.nc_t_student.measurements.centralMoments(3, lambda, n, loc, scale);
                    const std = Math.sqrt(µ2 - µ1 ** 2);
                    return central_µ3 / std ** 3;
                },
                kurtosis: function (lambda, n, loc, scale) {
                    const µ1 = dists.nc_t_student.measurements.nonCentralMoments(1, lambda, n, loc, scale);
                    const µ2 = dists.nc_t_student.measurements.nonCentralMoments(2, lambda, n, loc, scale);
                    const central_µ4 = dists.nc_t_student.measurements.centralMoments(4, lambda, n, loc, scale);
                    const std = Math.sqrt(µ2 - µ1 ** 2);
                    return central_µ4 / std ** 4;
                },
                median: function (lambda, n, loc, scale) {
                    return dists.nc_t_student.measurements.ppf(0.5, lambda, n, loc, scale);
                },
                mode: function (lambda, n, loc, scale) {
                    return undefined;
                },
            },
        },
    },
    normal: {
        measurements: {
            nonCentralMoments: function (k, miu, sigma) {
                return undefined;
            },
            centralMoments: function (k, miu, sigma) {
                return undefined;
            },
            stats: {
                mean: function (miu, sigma) {
                    return miu;
                },
                variance: function (miu, sigma) {
                    return sigma * 3;
                },
                standardDeviation: function (miu, sigma) {
                    return Math.sqrt(this.variance(miu, sigma));
                },
                skewness: function (miu, sigma) {
                    return 0;
                },
                kurtosis: function (miu, sigma) {
                    return 3;
                },
                median: function (miu, sigma) {
                    return dists.normal.measurements.ppf(0.5, miu, sigma);
                },
                mode: function (miu, sigma) {
                    return miu;
                },
            },
        },
    },
    pareto_first_kind: {
        measurements: {
            nonCentralMoments: function (k, xm, alpha, loc) {
                return (alpha * xm ** k) / (alpha - k);
            },
            centralMoments: function (k, xm, alpha, loc) {
                const µ1 = this.nonCentralMoments(1, xm, alpha, loc);
                const µ2 = this.nonCentralMoments(2, xm, alpha, loc);
                const µ3 = this.nonCentralMoments(3, xm, alpha, loc);
                const µ4 = this.nonCentralMoments(4, xm, alpha, loc);

                let result;
                switch (k) {
                    case 1:
                        result = 0;
                        break;
                    case 2:
                        result = µ2 - µ1 ** 2;
                        break;
                    case 3:
                        result = µ3 - 3 * µ1 * µ2 + 2 * µ1 ** 3;
                        break;
                    case 4:
                        result = µ4 - 4 * µ1 * µ3 + 6 * µ1 ** 2 * µ2 - 3 * µ1 ** 4;
                        break;
                }
                return result;
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
                    return central_µ3 / this.standardDeviation(xm, alpha, loc) ** 3;
                },
                kurtosis: function (xm, alpha, loc) {
                    const central_µ4 = dists.pareto_first_kind.measurements.centralMoments(4, xm, alpha, loc);
                    return central_µ4 / this.standardDeviation(xm, alpha, loc) ** 4;
                },
                median: function (xm, alpha, loc) {
                    return dists.pareto_first_kind.measurements.ppf(0.5, xm, alpha, loc);
                },
                mode: function (xm, alpha, loc) {
                    return xm + loc;
                },
            },
        },
    },
    pareto_second_kind: {
        measurements: {
            nonCentralMoments: function (k, xm, alpha, loc) {
                return (xm ** k * jStat.gammafn(alpha - k) * jStat.gammafn(1 + k)) / jStat.gammafn(alpha);
            },
            centralMoments: function (k, xm, alpha, loc) {
                const µ1 = this.nonCentralMoments(1, xm, alpha, loc);
                const µ2 = this.nonCentralMoments(2, xm, alpha, loc);
                const µ3 = this.nonCentralMoments(3, xm, alpha, loc);
                const µ4 = this.nonCentralMoments(4, xm, alpha, loc);

                let result;
                switch (k) {
                    case 1:
                        result = 0;
                        break;
                    case 2:
                        result = µ2 - µ1 ** 2;
                        break;
                    case 3:
                        result = µ3 - 3 * µ1 * µ2 + 2 * µ1 ** 3;
                        break;
                    case 4:
                        result = µ4 - 4 * µ1 * µ3 + 6 * µ1 ** 2 * µ2 - 3 * µ1 ** 4;
                        break;
                }
                return result;
            },
            stats: {
                mean: function (xm, alpha, loc) {
                    const µ1 = dists.pareto_second_kind.measurements.nonCentralMoments(1, xm, alpha, loc);
                    return loc + µ1;
                },
                variance: function (xm, alpha, loc) {
                    const µ1 = dists.pareto_second_kind.measurements.nonCentralMoments(1, xm, alpha, loc);
                    const µ2 = dists.pareto_second_kind.measurements.nonCentralMoments(2, xm, alpha, loc);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (xm, alpha, loc) {
                    return Math.sqrt(this.variance(xm, alpha, loc));
                },
                skewness: function (xm, alpha, loc) {
                    const central_µ3 = dists.pareto_second_kind.measurements.centralMoments(3, xm, alpha, loc);
                    return central_µ3 / this.standardDeviation(xm, alpha, loc) ** 3;
                },
                kurtosis: function (xm, alpha, loc) {
                    const central_µ4 = dists.pareto_second_kind.measurements.centralMoments(4, xm, alpha, loc);
                    return central_µ4 / this.standardDeviation(xm, alpha, loc) ** 4;
                },
                median: function (xm, alpha, loc) {
                    return dists.pareto_second_kind.measurements.ppf(0.5, xm, alpha, loc);
                },
                mode: function (xm, alpha, loc) {
                    return loc;
                },
            },
        },
    },
    pert: {
        measurements: {
            nonCentralMoments: function (k, a, b, c) {
                return undefined;
            },
            centralMoments: function (k, a, b, c) {
                return undefined;
            },
            stats: {
                mean: function (a, b, c) {
                    return (a + 4 * b + c) / 6;
                },
                variance: function (a, b, c) {
                    return ((this.mean(a, b, c) - a) * (c - this.mean(a, b, c))) / 7;
                },
                standardDeviation: function (a, b, c) {
                    return Math.sqrt(this.variance(a, b, c));
                },
                skewness: function (a, b, c) {
                    const α1 = (4 * b + c - 5 * a) / (c - a);
                    const α2 = (5 * c - a - 4 * b) / (c - a);
                    return (2 * (α2 - α1) * Math.sqrt(α1 + α2 + 1)) / ((α1 + α2 + 2) * Math.sqrt(α1 * α2));
                },
                kurtosis: function (a, b, c) {
                    const α1 = (4 * b + c - 5 * a) / (c - a);
                    const α2 = (5 * c - a - 4 * b) / (c - a);
                    return (6 * ((α2 - α1) ** 2 * (α1 + α2 + 1) - α1 * α2 * (α1 + α2 + 2))) / (α1 * α2 * (α1 + α2 + 2) * (α1 + α2 + 3)) + 3;
                },
                median: function (a, b, c) {
                    return dists.pert.measurements.ppf(0.5, a, b, c);
                },
                mode: function (a, b, c) {
                    return b;
                },
            },
        },
    },
    power_function: {
        measurements: {
            nonCentralMoments: function (k, alpha, a, b) {
                let result;
                switch (k) {
                    case 1:
                        result = (a + b * alpha) / (alpha + 1);
                        break;
                    case 2:
                        result = (2 * a ** 2 + 2 * alpha * a * b + alpha * (alpha + 1) * b ** 2) / ((alpha + 1) * (alpha + 2));
                        break;
                    case 3:
                        result =
                            (6 * a ** 3 + 6 * a ** 2 * b * alpha + 3 * a * b ** 2 * alpha * (1 + alpha) + b ** 3 * alpha * (1 + alpha) * (2 + alpha)) /
                            ((1 + alpha) * (2 + alpha) * (3 + alpha));
                        break;
                    case 4:
                        result =
                            (24 * a ** 4 +
                                24 * alpha * a ** 3 * b +
                                12 * alpha * (alpha + 1) * a ** 2 * b ** 2 +
                                4 * alpha * (alpha + 1) * (alpha + 2) * a * b ** 3 +
                                alpha * (alpha + 1) * (alpha + 2) * (alpha + 3) * b ** 4) /
                            ((alpha + 1) * (alpha + 2) * (alpha + 3) * (alpha + 4));
                        break;
                }
                return result;
            },
            centralMoments: function (k, alpha, a, b) {
                const µ1 = this.nonCentralMoments(1, alpha, a, b);
                const µ2 = this.nonCentralMoments(2, alpha, a, b);
                const µ3 = this.nonCentralMoments(3, alpha, a, b);
                const µ4 = this.nonCentralMoments(4, alpha, a, b);

                let result;
                switch (k) {
                    case 1:
                        result = 0;
                        break;
                    case 2:
                        result = µ2 - µ1 ** 2;
                        break;
                    case 3:
                        result = µ3 - 3 * µ1 * µ2 + 2 * µ1 ** 3;
                        break;
                    case 4:
                        result = µ4 - 4 * µ1 * µ3 + 6 * µ1 ** 2 * µ2 - 3 * µ1 ** 4;
                        break;
                }
                return result;
            },
            stats: {
                mean: function (alpha, a, b) {
                    const µ1 = dists.power_function.measurements.nonCentralMoments(1, alpha, a, b);
                    return µ1;
                },
                variance: function (alpha, a, b) {
                    const µ1 = dists.power_function.measurements.nonCentralMoments(1, alpha, a, b);
                    const µ2 = dists.power_function.measurements.nonCentralMoments(2, alpha, a, b);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (alpha, a, b) {
                    return Math.sqrt(this.variance(alpha, a, b));
                },
                skewness: function (alpha, a, b) {
                    const central_µ3 = dists.power_function.measurements.centralMoments(3, alpha, a, b);
                    return central_µ3 / this.standardDeviation(alpha, a, b) ** 3;
                },
                kurtosis: function (alpha, a, b) {
                    const central_µ4 = dists.power_function.measurements.centralMoments(4, alpha, a, b);
                    return central_µ4 / this.standardDeviation(alpha, a, b) ** 4;
                },
                median: function (alpha, a, b) {
                    return dists.power_function.measurements.ppf(0.5, alpha, a, b);
                },
                mode: function (alpha, a, b) {
                    return Math.max(a, b);
                },
            },
        },
    },
    rayleigh: {
        measurements: {
            nonCentralMoments: function (k, gamma, sigma) {
                return undefined;
            },
            centralMoments: function (k, gamma, sigma) {
                return undefined;
            },
            stats: {
                mean: function (gamma, sigma) {
                    return sigma * Math.sqrt(Math.PI / 2) + gamma;
                },
                variance: function (gamma, sigma) {
                    return sigma * sigma * (2 - Math.PI / 2);
                },
                standardDeviation: function (gamma, sigma) {
                    return Math.sqrt(this.variance(gamma, sigma));
                },
                skewness: function (gamma, sigma) {
                    return 0.6311;
                },
                kurtosis: function (gamma, sigma) {
                    return (24 * Math.PI - 6 * Math.PI * Math.PI - 16) / ((4 - Math.PI) * (4 - Math.PI)) + 3;
                },
                median: function (gamma, sigma) {
                    return dists.rayleigh.measurements.ppf(0.5, gamma, sigma);
                },
                mode: function (gamma, sigma) {
                    return gamma + sigma;
                },
            },
        },
    },
    reciprocal: {
        measurements: {
            nonCentralMoments: function (k, a, b) {
                return (b ** k - a ** k) / (k * (Math.log(b) - Math.log(a)));
            },
            centralMoments: function (k, a, b) {
                const µ1 = this.nonCentralMoments(1, a, b);
                const µ2 = this.nonCentralMoments(2, a, b);
                const µ3 = this.nonCentralMoments(3, a, b);
                const µ4 = this.nonCentralMoments(4, a, b);

                let result;
                switch (k) {
                    case 1:
                        result = 0;
                        break;
                    case 2:
                        result = µ2 - µ1 ** 2;
                        break;
                    case 3:
                        result = µ3 - 3 * µ1 * µ2 + 2 * µ1 ** 3;
                        break;
                    case 4:
                        result = µ4 - 4 * µ1 * µ3 + 6 * µ1 ** 2 * µ2 - 3 * µ1 ** 4;
                        break;
                }
                return result;
            },
            stats: {
                mean: function (a, b) {
                    const µ1 = dists.reciprocal.measurements.nonCentralMoments(1, a, b);
                    return µ1;
                },
                variance: function (a, b) {
                    const µ1 = dists.reciprocal.measurements.nonCentralMoments(1, a, b);
                    const µ2 = dists.reciprocal.measurements.nonCentralMoments(2, a, b);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (a, b) {
                    return Math.sqrt(this.variance(a, b));
                },
                skewness: function (a, b) {
                    const central_µ3 = dists.reciprocal.measurements.centralMoments(3, a, b);
                    return central_µ3 / this.standardDeviation(a, b) ** 3;
                },
                kurtosis: function (a, b) {
                    const central_µ4 = dists.reciprocal.measurements.centralMoments(4, a, b);
                    return central_µ4 / this.standardDeviation(a, b) ** 4;
                },
                median: function (a, b) {
                    return dists.reciprocal.measurements.ppf(0.5, a, b);
                },
                mode: function (a, b) {
                    return a;
                },
            },
        },
    },
    rice: {
        measurements: {
            nonCentralMoments: function (k, v, sigma) {
                let result;
                switch (k) {
                    case 1:
                        result =
                            sigma *
                            Math.sqrt(Math.PI / 2) *
                            Math.exp((-v * v) / (2 * sigma * sigma) / 2) *
                            ((1 - (-v * v) / (2 * sigma * sigma)) * BESSEL.besseli((-v * v) / (4 * sigma * sigma), 0) +
                                ((-v * v) / (2 * sigma * sigma)) * BESSEL.besseli((-v * v) / (4 * sigma * sigma), 1));
                        break;
                    case 2:
                        result = 2 * sigma * sigma + v * v;
                        break;
                    case 3:
                        result =
                            (3 *
                                sigma ** 3 *
                                Math.sqrt(Math.PI / 2) *
                                Math.exp((-v * v) / (2 * sigma * sigma) / 2) *
                                ((2 * ((-v * v) / (2 * sigma * sigma)) ** 2 - 6 * ((-v * v) / (2 * sigma * sigma)) + 3) * BESSEL.besseli((-v * v) / (4 * sigma * sigma), 0) -
                                    2 * ((-v * v) / (2 * sigma * sigma) - 2) * ((-v * v) / (2 * sigma * sigma)) * BESSEL.besseli((-v * v) / (4 * sigma * sigma), 1))) /
                            3;
                        break;
                    case 4:
                        result = 8 * sigma ** 4 + 8 * sigma * sigma * v * v + v ** 4;
                        break;
                }
                return result;
            },
            centralMoments: function (k, v, sigma) {
                const µ1 = this.nonCentralMoments(1, v, sigma);
                const µ2 = this.nonCentralMoments(2, v, sigma);
                const µ3 = this.nonCentralMoments(3, v, sigma);
                const µ4 = this.nonCentralMoments(4, v, sigma);

                let result;
                switch (k) {
                    case 1:
                        result = 0;
                        break;
                    case 2:
                        result = µ2 - µ1 ** 2;
                        break;
                    case 3:
                        result = µ3 - 3 * µ1 * µ2 + 2 * µ1 ** 3;
                        break;
                    case 4:
                        result = µ4 - 4 * µ1 * µ3 + 6 * µ1 ** 2 * µ2 - 3 * µ1 ** 4;
                        break;
                }
                return result;
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
                    return central_µ3 / this.standardDeviation(v, sigma) ** 3;
                },
                kurtosis: function (v, sigma) {
                    const central_µ4 = dists.rice.measurements.centralMoments(4, v, sigma);
                    return central_µ4 / this.standardDeviation(v, sigma) ** 4;
                },
                median: function (v, sigma) {
                    return dists.rice.measurements.ppf(0.5, v, sigma);
                },
                mode: function (v, sigma) {
                    return undefined;
                },
            },
        },
    },
    semicircular: {
        measurements: {
            nonCentralMoments: function (k, loc, R) {
                return undefined;
            },
            centralMoments: function (k, loc, R) {
                return undefined;
            },
            stats: {
                mean: function (loc, R) {
                    return loc;
                },
                variance: function (loc, R) {
                    return (R * R) / 4;
                },
                standardDeviation: function (loc, R) {
                    return Math.sqrt(this.variance(loc, R));
                },
                skewness: function (loc, R) {
                    return 0;
                },
                kurtosis: function (loc, R) {
                    return 2;
                },
                median: function (loc, R) {
                    return dists.semicircular.measurements.ppf(0.5, loc, R);
                },
                mode: function (loc, R) {
                    return loc;
                },
            },
        },
    },
    trapezoidal: {
        measurements: {
            nonCentralMoments: function (k, a, b, c, d) {
                return (2 / (d + c - b - a)) * (1 / ((k + 1) * (k + 2))) * ((d ** (k + 2) - c ** (k + 2)) / (d - c) - (b ** (k + 2) - a ** (k + 2)) / (b - a));
            },
            centralMoments: function (k, a, b, c, d) {
                const µ1 = this.nonCentralMoments(1, a, b, c, d);
                const µ2 = this.nonCentralMoments(2, a, b, c, d);
                const µ3 = this.nonCentralMoments(3, a, b, c, d);
                const µ4 = this.nonCentralMoments(4, a, b, c, d);

                let result;
                switch (k) {
                    case 1:
                        result = 0;
                        break;
                    case 2:
                        result = µ2 - µ1 ** 2;
                        break;
                    case 3:
                        result = µ3 - 3 * µ1 * µ2 + 2 * µ1 ** 3;
                        break;
                    case 4:
                        result = µ4 - 4 * µ1 * µ3 + 6 * µ1 ** 2 * µ2 - 3 * µ1 ** 4;
                        break;
                }
                return result;
            },
            stats: {
                mean: function (a, b, c, d) {
                    const µ1 = dists.trapezoidal.measurements.nonCentralMoments(1, a, b, c, d);
                    return µ1;
                },
                variance: function (a, b, c, d) {
                    const µ1 = dists.trapezoidal.measurements.nonCentralMoments(1, a, b, c, d);
                    const µ2 = dists.trapezoidal.measurements.nonCentralMoments(2, a, b, c, d);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (a, b, c, d) {
                    return Math.sqrt(this.variance(a, b, c, d));
                },
                skewness: function (a, b, c, d) {
                    const central_µ3 = dists.trapezoidal.measurements.centralMoments(3, a, b, c, d);
                    return central_µ3 / this.standardDeviation(a, b, c, d) ** 3;
                },
                kurtosis: function (a, b, c, d) {
                    const central_µ4 = dists.trapezoidal.measurements.centralMoments(4, a, b, c, d);
                    return central_µ4 / this.standardDeviation(a, b, c, d) ** 4;
                },
                median: function (a, b, c, d) {
                    return dists.trapezoidal.measurements.ppf(0.5, a, b, c, d);
                },
                mode: function (a, b, c, d) {
                    return undefined;
                },
            },
        },
    },
    triangular: {
        measurements: {
            nonCentralMoments: function (k, a, b, c) {
                return undefined;
            },
            centralMoments: function (k, a, b, c) {
                return undefined;
            },
            stats: {
                mean: function (a, b, c) {
                    return (a + b + c) / 3;
                },
                variance: function (a, b, c) {
                    return (a ** 2 + b ** 2 + c ** 2 - a * b - a * c - b * c) / 18;
                },
                standardDeviation: function (a, b, c) {
                    return Math.sqrt(this.variance(a, b, c));
                },
                skewness: function (a, b, c) {
                    return (Math.sqrt(2) * (a + b - 2 * c) * (2 * a - b - c) * (a - 2 * b + c)) / (5 * (a ** 2 + b ** 2 + c ** 2 - a * b - a * c - b * c) ** (3 / 2));
                },
                kurtosis: function (a, b, c) {
                    return 3 - 3 / 5;
                },
                median: function (a, b, c) {
                    return dists.triangular.measurements.ppf(0.5, a, b, c);
                },
                mode: function (a, b, c) {
                    return c;
                },
            },
        },
    },
    t_student: {
        measurements: {
            nonCentralMoments: function (k, df) {
                return undefined;
            },
            centralMoments: function (k, df) {
                return undefined;
            },
            stats: {
                mean: function (df) {
                    return 0;
                },
                variance: function (df) {
                    return df / (df - 2);
                },
                standardDeviation: function (df) {
                    return Math.sqrt(this.variance(df));
                },
                skewness: function (df) {
                    return 0;
                },
                kurtosis: function (df) {
                    return 6 / (df - 4) + 3;
                },
                median: function (df) {
                    return dists.t_student.measurements.ppf(0.5, df);
                },
                mode: function (df) {
                    return 0;
                },
            },
        },
    },
    t_student_3p: {
        measurements: {
            nonCentralMoments: function (k, df, loc, scale) {
                return undefined;
            },
            centralMoments: function (k, df, loc, scale) {
                return undefined;
            },
            stats: {
                mean: function (df, loc, scale) {
                    return loc;
                },
                variance: function (df, loc, scale) {
                    return (scale * scale * df) / (df - 2);
                },
                standardDeviation: function (df, loc, scale) {
                    return Math.sqrt(this.variance(df, loc, scale));
                },
                skewness: function (df, loc, scale) {
                    return 0;
                },
                kurtosis: function (df, loc, scale) {
                    return 6 / (df - 4) + 3;
                },
                median: function (df, loc, scale) {
                    return dists.t_student_3p.measurements.ppf(0.5, df, loc, scale);
                },
                mode: function (df, loc, scale) {
                    return loc;
                },
            },
        },
    },
    uniform: {
        measurements: {
            nonCentralMoments: function (k, a, b) {
                return undefined;
            },
            centralMoments: function (k, a, b) {
                return undefined;
            },
            stats: {
                mean: function (a, b) {
                    return (a + b) / 2;
                },
                variance: function (a, b) {
                    return (b - a) ** 2 / 12;
                },
                standardDeviation: function (a, b) {
                    return Math.sqrt(this.variance(a, b));
                },
                skewness: function (a, b) {
                    return 0;
                },
                kurtosis: function (a, b) {
                    return 3 - 6 / 5;
                },
                median: function (a, b) {
                    return dists.uniform.measurements.ppf(0.5, a, b);
                },
                mode: function (a, b) {
                    return undefined;
                },
            },
        },
    },
    weibull: {
        measurements: {
            nonCentralMoments: function (k, alpha, beta) {
                return beta ** k * jStat.gammafn(1 + k / alpha);
            },
            centralMoments: function (k, alpha, beta) {
                const µ1 = this.nonCentralMoments(1, alpha, beta);
                const µ2 = this.nonCentralMoments(2, alpha, beta);
                const µ3 = this.nonCentralMoments(3, alpha, beta);
                const µ4 = this.nonCentralMoments(4, alpha, beta);

                let result;
                switch (k) {
                    case 1:
                        result = 0;
                        break;
                    case 2:
                        result = µ2 - µ1 ** 2;
                        break;
                    case 3:
                        result = µ3 - 3 * µ1 * µ2 + 2 * µ1 ** 3;
                        break;
                    case 4:
                        result = µ4 - 4 * µ1 * µ3 + 6 * µ1 ** 2 * µ2 - 3 * µ1 ** 4;
                        break;
                }
                return result;
            },
            stats: {
                mean: function (alpha, beta) {
                    const µ1 = dists.weibull.measurements.nonCentralMoments(1, alpha, beta);
                    return µ1;
                },
                variance: function (alpha, beta) {
                    const µ1 = dists.weibull.measurements.nonCentralMoments(1, alpha, beta);
                    const µ2 = dists.weibull.measurements.nonCentralMoments(2, alpha, beta);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (alpha, beta) {
                    return Math.sqrt(this.variance(alpha, beta));
                },
                skewness: function (alpha, beta) {
                    const central_µ3 = dists.weibull.measurements.centralMoments(3, alpha, beta);
                    return central_µ3 / this.standardDeviation(alpha, beta) ** 3;
                },
                kurtosis: function (alpha, beta) {
                    const central_µ4 = dists.weibull.measurements.centralMoments(4, alpha, beta);
                    return central_µ4 / this.standardDeviation(alpha, beta) ** 4;
                },
                median: function (alpha, beta) {
                    return dists.weibull.measurements.ppf(0.5, alpha, beta);
                },
                mode: function (alpha, beta) {
                    return beta * ((alpha - 1) / alpha) ** (1 / alpha);
                },
            },
        },
    },
    weibull_3p: {
        measurements: {
            nonCentralMoments: function (k, alpha, beta, loc) {
                return beta ** k * jStat.gammafn(1 + k / alpha);
            },
            centralMoments: function (k, alpha, beta, loc) {
                const µ1 = this.nonCentralMoments(1, alpha, beta, loc);
                const µ2 = this.nonCentralMoments(2, alpha, beta, loc);
                const µ3 = this.nonCentralMoments(3, alpha, beta, loc);
                const µ4 = this.nonCentralMoments(4, alpha, beta, loc);

                let result;
                switch (k) {
                    case 1:
                        result = 0;
                        break;
                    case 2:
                        result = µ2 - µ1 ** 2;
                        break;
                    case 3:
                        result = µ3 - 3 * µ1 * µ2 + 2 * µ1 ** 3;
                        break;
                    case 4:
                        result = µ4 - 4 * µ1 * µ3 + 6 * µ1 ** 2 * µ2 - 3 * µ1 ** 4;
                        break;
                }
                return result;
            },
            stats: {
                mean: function (alpha, beta, loc) {
                    const µ1 = dists.weibull_3p.measurements.nonCentralMoments(1, alpha, beta, loc);
                    return loc + µ1;
                },
                variance: function (alpha, beta, loc) {
                    const µ1 = dists.weibull_3p.measurements.nonCentralMoments(1, alpha, beta, loc);
                    const µ2 = dists.weibull_3p.measurements.nonCentralMoments(2, alpha, beta, loc);
                    return µ2 - µ1 ** 2;
                },
                standardDeviation: function (alpha, beta, loc) {
                    return Math.sqrt(this.variance(alpha, beta, loc));
                },
                skewness: function (alpha, beta, loc) {
                    const central_µ3 = dists.weibull_3p.measurements.centralMoments(3, alpha, beta, loc);
                    return central_µ3 / this.standardDeviation(alpha, beta, loc) ** 3;
                },
                kurtosis: function (alpha, beta, loc) {
                    const central_µ4 = dists.weibull_3p.measurements.centralMoments(4, alpha, beta, loc);
                    return central_µ4 / this.standardDeviation(alpha, beta, loc) ** 4;
                },
                median: function (alpha, beta, loc) {
                    return dists.weibull_3p.measurements.ppf(0.5, alpha, beta, loc);
                },
                mode: function (alpha, beta, loc) {
                    return beta * ((alpha - 1) / alpha) ** (1 / alpha) + loc;
                },
            },
        },
    },
};
