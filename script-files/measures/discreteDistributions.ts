import { jStat } from "./dependencies/jstat";
import stdDists from "@stdlib/stats/base/dists";
import type { DiscreteDistributions } from "@/interfaces/DiscreteDistributions";
import { nCr } from "./dependencies/nCr";

const discreteDistributions: DiscreteDistributions = {
    bernoulli: {
        cdf: function (x, p) {
            if (x < 0) {
                return 0;
            } else if (x >= 0 && x < 1) {
                return 1 - p;
            } else {
                return 1;
            }
        },
        pmf: function (x, p) {
            return p ** x * (1 - p) ** (1 - x);
        },
        ppf: function (u, p) {
            if (u <= p) {
                return 1;
            } else {
                return 0;
            }
        },
        sample: function (n_samples, p) {
            return [...Array(n_samples)].map(() => this.ppf(Math.random(), p));
        },
        parametersRestrictions(p) {
            const v1 = p > 0 && p < 1;
            return v1;
        },
    },
    binomial: {
        cdf: function (x, n, p) {
            return jStat.ibeta(1 - p, n - x, x + 1);
        },
        pmf: function (x, n, p) {
            return nCr(n, x) * Math.pow(p, x) * Math.pow(1 - p, n - x);
        },
        ppf: function (u, n, p) {
            return stdDists.binomial.quantile(u, n, p);
        },
        sample: function (n_samples, n, p) {
            return [...Array(n_samples)].map(() => this.ppf(Math.random(), n, p));
        },
        parametersRestrictions(n, p) {
            const v1 = p > 0 && p < 1;
            const v2 = n > 0;
            const v3 = Number.isInteger(n);
            return v1 && v2 && v3;
        },
    },
    geometric: {
        cdf: function (x, p) {
            return 1 - (1 - p) ** x;
        },
        pmf: function (x, p) {
            return p * (1 - p) ** (x - 1);
        },
        ppf: function (u, p) {
            return Math.ceil(numpy.log(1 - u) / numpy.log(1 - p));
        },
        sample: function (n_samples, p) {
            return [...Array(n_samples)].map(() => this.ppf(Math.random(), p));
        },
        parametersRestrictions(p) {
            const v1 = p > 0 && p < 1;
            return v1;
        },
    },
    hypergeometric: {
        cdf: function (x, N, K, n) {
            return stdDists.hypergeometric.cdf(x, N, K, n);
        },
        pmf: function (x, N, K, n) {
            return (nCr(K, x) * nCr(N - K, n - x)) / nCr(N, n);
        },
        ppf: function (u, N, K, n) {
            function _hypergeometricCdf(x: number, N: number, K: number, n: number): number {
                return stdDists.hypergeometric.cdf(x, N, K, n);
            }
            let x = Math.min(0, n + K - N);
            let prob;
            let condition = true;
            while (condition) {
                prob = _hypergeometricCdf(x, N, K, n);
                if (prob > u) {
                    break;
                }
                x += 1;
            }
            return stdDists.hypergeometric.quantile(u, N, K, n);
        },
        sample: function (n_samples, N, K, n) {
            return [...Array(n_samples)].map(() => this.ppf(Math.random(), N, K, n));
        },
        parametersRestrictions(N, K, n) {
            const v1 = N > 0 && Number.isInteger(N);
            const v2 = K > 0 && Number.isInteger(K);
            const v3 = n > 0 && Number.isInteger(n);
            return v1 && v2 && v3;
        },
    },
    logarithmic: {
        cdf: function (x, p) {
            let r = 0;
            for (let t = 1; t <= x; t++) {
                r = r + this.pmf(t, p);
            }
            return r;
        },
        pmf: function (x, p) {
            return -(p ** x) / (numpy.log(1 - p) * x);
        },
        ppf: function (u, p) {
            function _logarithmicCdf(x: number, p: number) {
                const _logarithmicPmf = (t: number, p: number): number => -(p ** t) / (numpy.log(1 - p) * t);
                let r = 0;
                for (let t = 1; t <= x; t++) {
                    r = r + _logarithmicPmf(t, p);
                }
                return r;
            }
            let x = 1;
            let prob;
            let condition = true;
            while (condition) {
                prob = _logarithmicCdf(x, p);
                if (prob > u) {
                    break;
                }
                x += 1;
            }
            return x;
        },
        sample: function (n_samples, p) {
            return [...Array(n_samples)].map(() => this.ppf(Math.random(), p));
        },
        parametersRestrictions(p) {
            const v1 = p > 0 && p < 1;
            return v1;
        },
    },
    negative_binomial: {
        cdf: function (x, r, p) {
            return jStat.ibeta(p, r, x + 1);
        },
        pmf: function (x, r, p) {
            return nCr(r + x - 1, x) * p ** r * (1 - p) ** x;
        },
        ppf: function (u, r, p) {
            return stdDists.negativeBinomial.quantile(u, r, p);
        },
        sample: function (n_samples, r, p) {
            return [...Array(n_samples)].map(() => this.ppf(Math.random(), r, p));
        },
        parametersRestrictions(r, p) {
            const v1 = p > 0 && p < 1;
            const v2 = r > 0;
            const v3 = Number.isInteger(r);
            return v1 && v2 && v3;
        },
    },
    poisson: {
        cdf: function (x, lambda) {
            return stdDists.poisson.cdf(x, lambda);
        },
        pmf: function (x, lambda) {
            return (lambda ** x * numpy.exp(-lambda)) / jStat.factorial(x);
        },
        ppf: function (u, lambda) {
            return stdDists.poisson.quantile(u, lambda);
        },
        sample: function (n_samples, lambda) {
            return [...Array(n_samples)].map(() => this.ppf(Math.random(), lambda));
        },
        parametersRestrictions(lambda) {
            const v1 = lambda > 0;
            return v1;
        },
    },
    uniform: {
        cdf: function (x, a, b) {
            return (x - a + 1) / (b - a + 1);
        },
        pmf: function (x, a, b) {
            return 1 / (b - a + 1);
        },
        ppf: function (u, a, b) {
            return Math.ceil(u * (b - a + 1) + a - 1);
        },
        sample: function (n_samples, a, b) {
            const r = [...Array(n_samples)].map(() => this.ppf(Math.random(), a, b));
            console.log(r);
            console.log(Math.max(...r), Math.min(...r));
            return r;
        },
        parametersRestrictions(a, b) {
            const v1 = b > a;
            const v2 = Number.isInteger(b);
            const v3 = Number.isInteger(a);
            return v1 && v2 && v3;
        },
    },
};

export { discreteDistributions };
