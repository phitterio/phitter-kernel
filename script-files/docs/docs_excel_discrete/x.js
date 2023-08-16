var { jStat } = require("jstat");
  
const dists = {
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
                    return A + (B - A) * (alpha) / (alpha + beta);
                },
                variance: function (alpha, beta, A, B) {
                    return (alpha * beta) / (Math.pow(alpha + beta, 2) * (alpha + beta + 1)) * ((B - A) ** 2);
                },
                standardDeviation: function (alpha, beta, A, B) {
                    return Math.sqrt(this.variance(alpha, beta, A, B));
                },
                skewness: function (alpha, beta, A, B) {
                    return (2 * (beta - alpha) * Math.sqrt(alpha + beta + 1)) / ((alpha + beta + 2) * Math.sqrt(alpha * beta));
                },
                kurtosis: function (alpha, beta, A, B) {
                    return 3 * ((alpha + beta + 1) * (2 * ((alpha + beta) ** 2) + alpha * beta * (alpha + beta - 6))) / (alpha * beta * (alpha + beta + 2) * (alpha + beta + 3))
                },
                median: function (alpha, beta, A, B) {
                    return dists.beta.ppf(alpha, beta, A, B);
                },
                mode: function (alpha, beta, A, B) {
                    if (alpha > 1 && beta > 1) {
                        return A + (B - A) * ((alpha - 1)) / (alpha + beta - 2);
                    }
                    return undefined;
                },
            },
        }
    },
    gamma: {
        measurements: {
            nonCentralMoments: function (k, alpha, beta) {
                return (beta ** k) * (jStat.gammafn(k + alpha) / (jStat.gammafn(alpha)));
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
                    return central_µ3 / (this.standardDeviation(alpha, beta) ** 3);
                },
                kurtosis: function (alpha, beta) {
                    const central_µ4 = dists.gamma.measurements.centralMoments(4, alpha, beta);
                    return central_µ4 / (this.standardDeviation(alpha, beta) ** 4);
                },
                median: function (alpha, beta) {
                    return dists.gamma.measurements.ppf(0.5, alpha, beta, A, B);
                },
                mode: function (alpha, beta) {
                    if (alpha > 1) {
                        return (alpha - 1) * beta;
                    }
                    return undefined;
                },
            },
        }
    },
    nc_f: {
        measurements: {
            nonCentralMoments: function (k, lambda, n1, n2) {
                let result;
                switch (k) {
                    case 1: result = (C5/C4)*((C4+C3)/(C5-2)); break;
                    case 2: result = ((C5/C4)^2*(1/((C5-2)*(C5-4)))*(C3^2+(2*C3+C4)*(C4+2))); break;
                    case 3: result = ((C5/C4)^3*(1/((C5-2)*(C5-4)*(C5-6)))*(C3^3+3*(C4+4)*C3^2+(3*C3+C4)*(C4+4)*(C4+2))); break;
                    case 4: result = ((C5/C4)^4*(1/((C5-2)*(C5-4)*(C5-6)*(C5-8)))*(C3^4+4*(C4+6)*C3^3+6*(C4+6)*(C4+4)*C3^2+(4*C3+C4)*(C4+2)*(C4+4)*(C4+6))); break;
                };
                return result
            },
            centralMoments: function (k, lambda, n1, n2) {
                const µ1 = this.nonCentralMoments(1, lambda, n1, n2);
                const µ2 = this.nonCentralMoments(2, lambda, n1, n2);
                const µ3 = this.nonCentralMoments(3, lambda, n1, n2);
                const µ4 = this.nonCentralMoments(4, lambda, n1, n2);

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
                    return central_µ3 / (this.standardDeviation(lambda, n1, n2) ** 3);
                },
                kurtosis: function (lambda, n1, n2) {
                    const central_µ4 = dists.nc_f.measurements.centralMoments(4, lambda, n1, n2);
                    return central_µ4 / (this.standardDeviation(lambda, n1, n2) ** 4);
                },
                median: function (lambda, n1, n2) {
                    return dists.nc_f.measurements.ppf(0.5, lambda, n1, n2, A, B);
                },
                mode: function (lambda, n1, n2) {
                    return undefined;
                },
            },
        }
    },
};

// console.log(dists.beta.measurements.stats.mean(5, 6, 100, 1000))
// console.log(dists.beta.measurements.stats.variance(5, 6, 100, 1000))
// console.log(dists.beta.measurements.stats.standardDeviation(5, 6, 100, 1000))
// console.log(dists.beta.measurements.stats.skewness(5, 6, 100, 1000))
// console.log(dists.beta.measurements.stats.kurtosis(5, 6, 100, 1000))
// console.log(dists.beta.measurements.stats.median(5, 6, 100, 1000))
// console.log(dists.beta.measurements.stats.mode(5, 6, 100, 1000))


console.log(dists.gamma.measurements.stats.mean(7, 5))
console.log(dists.gamma.measurements.stats.variance(7, 5))
console.log(dists.gamma.measurements.stats.standardDeviation(7, 5))
console.log(dists.gamma.measurements.stats.skewness(7, 5))
console.log(dists.gamma.measurements.stats.kurtosis(7, 5))
// console.log(dists.gamma.measurements.stats.median(7, 5))
console.log(dists.gamma.measurements.stats.mode(7, 5))


// measurements: {
//     nonCentralMoments: function (k, alpha, beta, A, B) {
//         return undefined;
//     },
//     centralMoments: function (k, alpha, beta, A, B) {
//         return undefined;
//     },
//     stats: {
//         mean: function () {
//             return ;
//         },
//         variane: function () {
//             return ;
//         },
//         standardDeviation: function () {
//             return ;
//         },
//         skewness: function () {
//             return ;
//         },
//         kurtosis: function () {
//             return ;
//         },
//         median: function () {
//             return ;
//         },
//         mode: function () {
//             return ;
//         },
//     },
// }


// measurements: {
//     nonCentralMoments: function (k, p) {
//         return ;
//     },
//     centralMoments: function (k, p) {
//         const µ1 = this.nonCentralMoments(1, p);
//         const µ2 = this.nonCentralMoments(2, p);
//         const µ3 = this.nonCentralMoments(3, p);
//         const µ4 = this.nonCentralMoments(4, p);

//         let result;
//         switch (k) {
//             case 1: result = 0; break;
//             case 2: result = µ2 - µ1 ** 2; break;
//             case 3: result = µ3 - 3 * µ1 * µ2 + 2 * µ1 ** 3; break;
//             case 4: result = µ4 - 4 * µ1 * µ3 + 6 * (µ1 ** 2) * µ2 - 3 * (µ1 ** 4); break;
//         };
//         return result
//     },
//     stats: {
//         mean: function (p) {
//             const µ1 = dists.gamma.measurements.nonCentralMoments(1, p);
//             return µ1;
//         },
//         variance: function (p) {
//             const µ1 = dists.gamma.measurements.nonCentralMoments(1, p);
//             const µ2 = dists.gamma.measurements.nonCentralMoments(2, p);
//             return µ2 - µ1 ** 2;
//         },
//         standardDeviation: function (p) {
//             return Math.sqrt(this.variance(p));
//         },
//         skewness: function (p) {
//             const central_µ3 = dists.gamma.measurements.centralMoments(3, p);
//             console.log(central_µ3)
//             return central_µ3 / (this.standardDeviation(p) ** 3);
//         },
//         kurtosis: function (p) {
//             const central_µ4 = dists.gamma.measurements.centralMoments(4, p);
//             return central_µ4 / (this.standardDeviation(p) ** 4);
//         },
//         median: function (p) {
//             return dists.gamma.measurements.ppf(0.5, p, A, B);
//         },
//         mode: function (p) {
//             return ;
//         },
//     },
// }

// =DISTR.BETA.N(x;a;b;VERDADERO)	                         jStat.ibeta(x,a,b)
// =INV.BETA.N(u;a;b)	                                     jStat.ibetainv(u,a,b)
// =EXP(GAMMA.LN(a)+GAMMA.LN(a)-GAMMA.LN(a+b))	             jStat.betafn(a,b)
// =DISTR.GAMMA.N(x;a;1;VERDADERO)	                         jStat.lowRegGamma(a,x)
// =INV.GAMMA(u;a;1)	                                     jStat.gammapinv(u, a)
// =GAMMA(x)	                                             jStat.gammafn(x)
// =DISTR.NORM.ESTAND.N(x;VERDADERO)	                     jStat.std_cdf(x)
// =DISTR.NORM.ESTAND.N(x;FALSO)	                         jStat.std_pdf(x)
// =INV.NORM.ESTAND(u)	                                     jStat.inv_std_cdf(u)
// =BESSELI(x;n)	                                         BESSEL.besseli(x,n)
// =COMBINAT(n;r)                                            nCr(n, r)


