import { jStat } from "./dependencies/jstat";
import { BESSEL } from "./dependencies/bessel";
import { usePyodideStore } from "@/stores/pyodideStore";
import type { ContinuousDistributions } from "@/interfaces/ContinuousDistributions";

const continuousDistributions: ContinuousDistributions = {
    alpha: {
        cdf: function (x, alpha, loc, scale) {
            const std_cdf = (t: number): number => 0.5 * (1 + jStat.erf(t / Math.sqrt(2)));
            const z = (t: number): number => (t - loc) / scale;
            return std_cdf(alpha - 1 / z(x)) / std_cdf(alpha);
        },
        pdf: function (x, alpha, loc, scale) {
            const std_cdf = (t: number): number => 0.5 * (1 + jStat.erf(t / Math.sqrt(2)));
            const z = (t: number): number => (t - loc) / scale;
            return (1 / (scale * z(x) * z(x) * std_cdf(alpha) * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * (alpha - 1 / z(x)) ** 2);
        },
        ppf: function (u, alpha, loc, scale) {
            const std_cdf = (t: number): number => 0.5 * (1 + jStat.erf(t / Math.sqrt(2)));
            const inv_std_cdf = (t: number): number => Math.sqrt(2) * jStat.erfinv(2 * t - 1);
            return loc + scale / (alpha - inv_std_cdf(u * std_cdf(alpha)));
        },
        sample: function (n_samples, alpha, loc, scale) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), alpha, loc, scale));
        },
        parametersRestrictions: function (alpha, loc, scale) {
            const v0 = Boolean(1 + 0 * loc);
            const v1 = alpha > 0;
            const v2 = scale > 0;
            return v0 && v1 && v2;
        },
    },
    arcsine: {
        cdf: function (x, a, b) {
            const z = (t: number): number => (t - a) / (b - a);
            return (2 * Math.asin(Math.sqrt(z(x)))) / Math.PI;
        },
        pdf: function (x, a, b) {
            return 1 / (Math.PI * Math.sqrt((x - a) * (b - x)));
        },
        ppf: function (u, a, b) {
            return a + (b - a) * Math.sin((u * Math.PI) / 2) ** 2;
        },
        sample: function (n_samples, a, b) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), a, b));
        },
        parametersRestrictions: function (a, b) {
            const v1 = b > a;
            return v1;
        },
    },
    argus: {
        cdf: function (x, chi, loc, scale) {
            const z = (t: number): number => (t - loc) / scale;
            const std_cdf = (t: number): number => 0.5 * (1 + jStat.erf(t / Math.sqrt(2)));
            const std_pdf = (t: number): number => (1 / Math.sqrt(2 * Math.PI)) * Math.exp(-(t ** 2 / 2));
            const Ψ = (t: number): number => std_cdf(t) - t * std_pdf(t) - 0.5;
            return 1 - Ψ(chi * Math.sqrt(1 - z(x) * z(x))) / Ψ(chi);
        },
        pdf: function (x, chi, loc, scale) {
            const z = (t: number): number => (t - loc) / scale;
            const std_cdf = (t: number): number => 0.5 * (1 + jStat.erf(t / Math.sqrt(2)));
            const std_pdf = (t: number): number => (1 / Math.sqrt(2 * Math.PI)) * Math.exp((-t * t) / 2);
            const Ψ = (t: number): number => std_cdf(t) - t * std_pdf(t) - 0.5;
            return (1 / scale) * (chi ** 3 / (Math.sqrt(2 * Math.PI) * Ψ(chi))) * z(x) * Math.sqrt(1 - z(x) * z(x)) * Math.exp(-0.5 * chi ** 2 * (1 - z(x) * z(x)));
        },
        ppf: function (u, chi, loc, scale) {
            const y1 = (1 - u) * jStat.lowRegGamma(1.5, (chi * chi) / 2);
            const y2 = (2 * jStat.gammapinv(y1, 1.5)) / (chi * chi);
            return loc + scale * Math.sqrt(1 - y2);
        },
        sample: function (n_samples, chi, loc, scale) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), chi, loc, scale));
        },
        parametersRestrictions: function (chi, loc, scale) {
            const v0 = Boolean(1 + 0 * scale);
            const v1 = chi > 0;
            const v2 = scale > 0;
            return v0 && v1 && v2;
        },
    },
    beta: {
        cdf: function (x, alpha, beta, A, B) {
            const z = (t: number): number => (t - A) / (B - A);
            return jStat.ibeta(z(x), alpha, beta);
        },
        pdf: function (x, alpha, beta, A, B) {
            const z = (t: number): number => (t - A) / (B - A);
            return (z(x) ** (alpha - 1) * (1 - z(x)) ** (beta - 1)) / jStat.betafn(alpha, beta) / (B - A);
        },
        ppf: function (u, alpha, beta, A, B) {
            return A + (B - A) * jStat.ibetainv(u, alpha, beta);
        },
        sample: function (n_samples, alpha, beta, A, B) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), alpha, beta, A, B));
        },
        parametersRestrictions: function (alpha, beta, A, B) {
            const v1 = alpha > 0;
            const v2 = beta > 0;
            const v3 = A < B;
            return v1 && v2 && v3;
        },
    },
    beta_prime: {
        cdf: function (x, alpha, beta) {
            return jStat.ibeta(x / (x + 1), alpha, beta);
        },
        pdf: function (x, alpha, beta) {
            return (x ** (alpha - 1) * (1 + x) ** (-alpha - beta)) / jStat.betafn(alpha, beta);
        },
        ppf: function (u, alpha, beta) {
            return jStat.ibetainv(u, alpha, beta) / (1 - jStat.ibetainv(u, alpha, beta));
        },
        sample: function (n_samples, alpha, beta) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), alpha, beta));
        },
        parametersRestrictions: function (alpha, beta) {
            const v1 = alpha > 0;
            const v2 = beta > 0;
            return v1 && v2;
        },
    },
    beta_prime_4p: {
        cdf: function (x, alpha, beta, loc, scale) {
            const z = (t: number): number => (t - loc) / scale;
            return jStat.ibeta(z(x) / (z(x) + 1), alpha, beta);
        },
        pdf: function (x, alpha, beta, loc, scale) {
            const z = (t: number): number => (t - loc) / scale;
            return ((1 / scale) * (z(x) ** (alpha - 1) * (1 + z(x)) ** (-alpha - beta))) / jStat.betafn(alpha, beta);
        },
        ppf: function (u, alpha, beta, loc, scale) {
            return loc + (scale * jStat.ibetainv(u, alpha, beta)) / (1 - jStat.ibetainv(u, alpha, beta));
        },
        sample: function (n_samples, alpha, beta, loc, scale) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), alpha, beta, loc, scale));
        },
        parametersRestrictions: function (alpha, beta, loc, scale) {
            const v0 = Boolean(1 + 0 * loc);
            const v1 = alpha > 0;
            const v2 = beta > 0;
            const v3 = scale > 0;
            return v0 && v1 && v2 && v3;
        },
    },
    bradford: {
        cdf: function (x, c, min, max) {
            return Math.log(1 + (c * (x - min)) / (max - min)) / Math.log(c + 1);
        },
        pdf: function (x, c, min, max) {
            return c / ((c * (x - min) + max - min) * Math.log(c + 1));
        },
        ppf: function (u, c, min, max): number {
            return min + ((Math.exp(u * Math.log(1 + c)) - 1) * (max - min)) / c;
        },
        sample: function (n_samples, c, min, max) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), c, min, max));
        },
        parametersRestrictions: function (c, min, max) {
            const v1 = c > 0;
            const v2 = max > min;
            return v1 && v2;
        },
    },
    burr: {
        cdf: function (x, A, B, C) {
            return 1 - (1 + (x / A) ** B) ** -C;
        },
        pdf: function (x, A, B, C) {
            return ((B * C) / A) * (x / A) ** (B - 1) * (1 + (x / A) ** B) ** (-C - 1);
        },
        ppf: function (u, A, B, C) {
            return A * ((1 - u) ** (-1 / C) - 1) ** (1 / B);
        },
        sample: function (n_samples, A, B, C) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), A, B, C));
        },
        parametersRestrictions: function (A, B, C) {
            const v0 = Boolean(1 + 0 * B);
            const v1 = A > 0;
            const v2 = C > 0;
            return v0 && v1 && v2;
        },
    },
    burr_4p: {
        cdf: function (x, A, B, C, loc) {
            return 1 - (1 + ((x - loc) / A) ** B) ** -C;
        },
        pdf: function (x, A, B, C, loc) {
            return ((B * C) / A) * ((x - loc) / A) ** (B - 1) * (1 + ((x - loc) / A) ** B) ** (-C - 1);
        },
        ppf: function (u, A, B, C, loc) {
            return A * ((1 - u) ** (-1 / C) - 1) ** (1 / B) + loc;
        },
        sample: function (n_samples, A, B, C, loc) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), A, B, C, loc));
        },
        parametersRestrictions: function (A, B, C, loc) {
            const v0 = Boolean(1 + 0 * loc * B);
            const v1 = A > 0;
            const v2 = C > 0;
            return v0 && v1 && v2;
        },
    },
    cauchy: {
        cdf: function (x, x0, gamma) {
            return (1 / Math.PI) * Math.atan((x - x0) / gamma) + 1 / 2;
        },
        pdf: function (x, x0, gamma) {
            return 1 / (Math.PI * gamma * (1 + ((x - x0) / gamma) ** 2));
        },
        ppf: function (u, x0, gamma) {
            return x0 + gamma * Math.tan(Math.PI * (u - 0.5));
        },
        sample: function (n_samples, x0, gamma) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), x0, gamma));
        },
        parametersRestrictions: function (x0, gamma) {
            const v0 = Boolean(1 + 0 * x0);
            const v1 = gamma > 0;
            return v0 && v1;
        },
    },
    chi_square: {
        cdf: function (x, df) {
            return jStat.lowRegGamma(df / 2, x / 2);
        },
        pdf: function (x, df) {
            return (1 / (2 ** (df / 2) * jStat.gammafn(df / 2))) * x ** (df / 2 - 1) * Math.exp(-x / 2);
        },
        ppf: function (u, df) {
            return 2 * jStat.gammapinv(u, df / 2);
        },
        sample: function (n_samples, df) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), df));
        },
        parametersRestrictions: function (df) {
            const v1 = df > 0;
            const v2 = Number.isInteger(df);
            return v1 && v2;
        },
    },
    chi_square_3p: {
        cdf: function (x, df, loc, scale) {
            const z = (t: number): number => (t - loc) / scale;
            return jStat.lowRegGamma(df / 2, z(x) / 2);
        },
        pdf: function (x, df, loc, scale) {
            const z = (t: number): number => (t - loc) / scale;
            return (1 / scale) * (1 / (2 ** (df / 2) * jStat.gammafn(df / 2))) * z(x) ** (df / 2 - 1) * Math.exp(-z(x) / 2);
        },
        ppf: function (u, df, loc, scale) {
            return 2 * scale * jStat.gammapinv(u, df / 2) + loc;
        },
        sample: function (n_samples, df, loc, scale) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), df, loc, scale));
        },
        parametersRestrictions: function (df, loc, scale) {
            const v0 = Boolean(1 + 0 * loc);
            const v1 = df > 0;
            const v2 = Number.isInteger(df);
            const v3 = scale > 0;
            return v0 && v1 && v2 && v3;
        },
    },
    dagum: {
        cdf: function (x, a, b, p) {
            return (1 + (x / b) ** -a) ** -p;
        },
        pdf: function (x, a, b, p) {
            return ((a * p) / x) * ((x / b) ** (a * p) / ((x / b) ** a + 1) ** (p + 1));
        },
        ppf: function (u, a, b, p) {
            return b * (u ** (-1 / p) - 1) ** (-1 / a);
        },
        sample: function (n_samples, a, b, p) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), a, b, p));
        },
        parametersRestrictions: function (a, b, p) {
            const v1 = p > 0;
            const v2 = a > 0;
            const v3 = b > 0;
            return v1 && v2 && v3;
        },
    },
    dagum_4p: {
        cdf: function (x, a, b, p, loc) {
            return (1 + ((x - loc) / b) ** -a) ** -p;
        },
        pdf: function (x, a, b, p, loc) {
            return ((a * p) / (x - loc)) * (((x - loc) / b) ** (a * p) / (((x - loc) / b) ** a + 1) ** (p + 1));
        },
        ppf: function (u, a, b, p, loc) {
            return b * (u ** (-1 / p) - 1) ** (-1 / a) + loc;
        },
        sample: function (n_samples, a, b, p, loc) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), a, b, p, loc));
        },
        parametersRestrictions: function (a, b, p, loc) {
            const v0 = Boolean(1 + 0 * loc);
            const v1 = p > 0;
            const v2 = a > 0;
            const v3 = b > 0;
            return v0 && v1 && v2 && v3;
        },
    },
    erlang: {
        cdf: function (x, k, beta) {
            return jStat.lowRegGamma(k, x / beta);
        },
        pdf: function (x, k, beta) {
            return (beta ** -k * x ** (k - 1) * Math.exp(-(x / beta))) / jStat.factorial(k - 1);
        },
        ppf: function (u, k, beta) {
            return beta * jStat.gammapinv(u, k);
        },
        sample: function (n_samples, k, beta) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), k, beta));
        },
        parametersRestrictions: function (k, beta) {
            const v1 = k > 0;
            const v2 = beta > 0;
            const v3 = Number.isInteger(k);
            return v1 && v2 && v3;
        },
    },
    erlang_3p: {
        cdf: function (x, k, beta, loc) {
            return jStat.lowRegGamma(k, (x - loc) / beta);
        },
        pdf: function (x, k, beta, loc) {
            return (beta ** -k * (x - loc) ** (k - 1) * Math.exp(-((x - loc) / beta))) / jStat.factorial(k - 1);
        },
        ppf: function (u, k, beta, loc) {
            return beta * jStat.gammapinv(u, k) + loc;
        },
        sample: function (n_samples, k, beta, loc) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), k, beta, loc));
        },
        parametersRestrictions: function (k, beta, loc) {
            const v0 = Boolean(1 + 0 * loc);
            const v1 = k > 0;
            const v2 = beta > 0;
            const v3 = Number.isInteger(k);
            return v0 && v1 && v2 && v3;
        },
    },
    error_function: {
        cdf: function (x, h) {
            const std_cdf = (t: number): number => 0.5 * (1 + jStat.erf(t / Math.sqrt(2)));
            return std_cdf(2 ** 0.5 * h * x);
        },
        pdf: function (x, h) {
            return (h * Math.exp(-1 * (-h * x) ** 2)) / Math.sqrt(Math.PI);
        },
        ppf: function (u, h) {
            const inv_std_cdf = (t: number): number => Math.sqrt(2) * jStat.erfinv(2 * t - 1);
            return inv_std_cdf(u) / (h * Math.sqrt(2));
        },
        sample: function (n_samples, h) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), h));
        },
        parametersRestrictions: function (h) {
            const v1 = h > 0;
            return v1;
        },
    },
    exponential: {
        cdf: function (x, lambda) {
            return 1 - Math.exp(-lambda * x);
        },
        pdf: function (x, lambda) {
            return lambda * Math.exp(-lambda * x);
        },
        ppf: function (u, lambda) {
            return -Math.log(1 - u) / lambda;
        },
        sample: function (n_samples, lambda) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), lambda));
        },
        parametersRestrictions: function (lambda) {
            const v1 = lambda > 0;
            return v1;
        },
    },
    exponential_2p: {
        cdf: function (x, lambda, loc) {
            return 1 - Math.exp(-lambda * (x - loc));
        },
        pdf: function (x, lambda, loc) {
            return lambda * Math.exp(-lambda * (x - loc));
        },
        ppf: function (u, lambda, loc) {
            return loc - Math.log(1 - u) / lambda;
        },
        sample: function (n_samples, lambda, loc) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), lambda, loc));
        },
        parametersRestrictions: function (lambda, loc) {
            const v0 = Boolean(1 + 0 * loc);
            const v1 = lambda > 0;
            return v0 && v1;
        },
    },
    f: {
        cdf: function (x, df1, df2) {
            return jStat.ibeta((x * df1) / (df1 * x + df2), df1 / 2, df2 / 2);
        },
        pdf: function (x, df1, df2) {
            return (1 / jStat.betafn(df1 / 2, df2 / 2)) * (df1 / df2) ** (df1 / 2) * x ** (df1 / 2 - 1) * (1 + (x * df1) / df2) ** ((-1 * (df1 + df2)) / 2);
        },
        ppf: function (u, df1, df2) {
            const t = jStat.ibetainv(u, df1 / 2, df2 / 2);
            return (df2 * t) / (df1 * (1 - t));
        },
        sample: function (n_samples, df1, df2) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), df1, df2));
        },
        parametersRestrictions: function (df1, df2) {
            const v1 = df1 > 0;
            const v2 = df2 > 0;
            return v1 && v2;
        },
    },
    f_4p: {
        cdf: function (x, df1, df2, loc, scale) {
            const z = (t: number): number => (t - loc) / scale;
            return jStat.ibeta((z(x) * df1) / (df1 * z(x) + df2), df1 / 2, df2 / 2);
        },
        pdf: function (x, df1, df2, loc, scale) {
            const z = (t: number): number => (t - loc) / scale;
            return (1 / scale) * (1 / jStat.betafn(df1 / 2, df2 / 2)) * (df1 / df2) ** (df1 / 2) * z(x) ** (df1 / 2 - 1) * (1 + (z(x) * df1) / df2) ** ((-1 * (df1 + df2)) / 2);
        },
        ppf: function (u, df1, df2, loc, scale) {
            const t = jStat.ibetainv(u, df1 / 2, df2 / 2);
            return loc + (scale * (df2 * t)) / (df1 * (1 - t));
        },
        sample: function (n_samples, df1, df2, loc, scale) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), df1, df2, loc, scale));
        },
        parametersRestrictions: function (df1, df2, loc, scale) {
            const v0 = Boolean(1 + 0 * loc);
            const v1 = df1 > 0;
            const v2 = df2 > 0;
            const v3 = scale > 0;
            return v0 && v1 && v2 && v3;
        },
    },
    fatigue_life: {
        cdf: function (x, gamma, loc, scale) {
            const z = (t: number): number => Math.sqrt((t - loc) / scale);
            const std_cdf = (t: number): number => 0.5 * (1 + jStat.erf(t / Math.sqrt(2)));
            return std_cdf((z(x) - 1 / z(x)) / gamma);
        },
        pdf: function (x, gamma, loc, scale) {
            const z = (t: number): number => Math.sqrt((t - loc) / scale);
            const std_pdf = (t: number): number => (1 / Math.sqrt(2 * Math.PI)) * Math.exp((-1 / 2) * t ** 2);
            return ((z(x) + 1 / z(x)) / (2 * gamma * (x - loc))) * std_pdf((z(x) - 1 / z(x)) / gamma);
        },
        ppf: function (u, gamma, loc, scale) {
            const inv_std_cdf = (t: number): number => Math.sqrt(2) * jStat.erfinv(2 * t - 1);
            return loc + (scale * (gamma * inv_std_cdf(u) + Math.sqrt((gamma * inv_std_cdf(u)) ** 2 + 4)) ** 2) / 4;
        },
        sample: function (n_samples, gamma, loc, scale) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), gamma, loc, scale));
        },
        parametersRestrictions: function (gamma, loc, scale) {
            const v0 = Boolean(1 + 0 * loc);
            const v1 = scale > 0;
            const v2 = gamma > 0;
            return v0 && v1 && v2;
        },
    },
    folded_normal: {
        cdf: function (x, miu, sigma) {
            const z1 = (t: number): number => (t + miu) / sigma;
            const z2 = (t: number): number => (t - miu) / sigma;
            return 0.5 * (jStat.erf(z1(x) / Math.sqrt(2)) + jStat.erf(z2(x) / Math.sqrt(2)));
        },
        pdf: function (x, miu, sigma) {
            return Math.sqrt(2 / (Math.PI * sigma ** 2)) * Math.exp(-(x ** 2 + miu ** 2) / (2 * sigma ** 2)) * Math.cosh((miu * x) / sigma ** 2);
        },
        ppf: function (u, miu, sigma) {
            const inv_std_cdf = (t: number): number => Math.sqrt(2) * jStat.erfinv(2 * t - 1);
            return Math.abs(inv_std_cdf(1 - u) * sigma + miu);
        },
        sample: function (n_samples, miu, sigma) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), miu, sigma));
        },
        parametersRestrictions: function (miu, sigma) {
            const v0 = Boolean(1 + 0 * miu);
            const v1 = sigma > 0;
            return v0 && v1;
        },
    },
    frechet: {
        cdf: function (x, alpha, loc, scale) {
            const z = (t: number): number => (t - loc) / scale;
            return Math.exp(-1 * z(x) ** -alpha);
        },
        pdf: function (x, alpha, loc, scale) {
            const z = (t: number): number => (t - loc) / scale;
            return (alpha / scale) * z(x) ** (-1 - alpha) * Math.exp(-1 * z(x) ** -alpha);
        },
        ppf: function (u, alpha, loc, scale) {
            return loc + scale * (-Math.log(u)) ** (-1 / alpha);
        },
        sample: function (n_samples, alpha, loc, scale) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), alpha, loc, scale));
        },
        parametersRestrictions: function (alpha, loc, scale) {
            const v0 = Boolean(1 + 0 * loc);
            const v1 = alpha > 0;
            const v2 = scale > 0;
            return v0 && v1 && v2;
        },
    },
    gamma: {
        cdf: function (x, alpha, beta) {
            return jStat.lowRegGamma(alpha, x / beta);
        },
        pdf: function (x, alpha, beta) {
            return (beta ** -alpha * x ** (alpha - 1) * Math.exp(-(x / beta))) / jStat.gammafn(alpha);
        },
        ppf: function (u, alpha, beta) {
            return beta * jStat.gammapinv(u, alpha);
        },
        sample: function (n_samples, alpha, beta) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), alpha, beta));
        },
        parametersRestrictions: function (alpha, beta) {
            const v1 = alpha > 0;
            const v2 = beta > 0;
            return v1 && v2;
        },
    },
    gamma_3p: {
        cdf: function (x, alpha, beta, loc) {
            return jStat.lowRegGamma(alpha, (x - loc) / beta);
        },
        pdf: function (x, alpha, beta, loc) {
            return (beta ** -alpha * (x - loc) ** (alpha - 1) * Math.exp(-((x - loc) / beta))) / jStat.gammafn(alpha);
        },
        ppf: function (u, alpha, beta, loc) {
            return beta * jStat.gammapinv(u, alpha) + loc;
        },
        sample: function (n_samples, alpha, beta, loc) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), alpha, beta, loc));
        },
        parametersRestrictions: function (alpha, beta, loc) {
            const v0 = Boolean(1 + 0 * loc);
            const v1 = alpha > 0;
            const v2 = beta > 0;
            return v0 && v1 && v2;
        },
    },
    generalized_extreme_value: {
        cdf: function (x, ξ, miu, sigma) {
            const z = (t: number): number => (t - miu) / sigma;
            if (ξ === 0) {
                return Math.exp(-Math.exp(-z(x)));
            } else {
                return Math.exp(-1 * (1 + ξ * z(x)) ** (-1 / ξ));
            }
        },
        pdf: function (x, ξ, miu, sigma) {
            const z = (t: number): number => (t - miu) / sigma;
            if (ξ === 0) {
                return (1 / sigma) * Math.exp(-z(x) - Math.exp(-z(x)));
            } else {
                return (1 / sigma) * Math.exp(-1 * (1 + ξ * z(x)) ** (-1 / ξ)) * (1 + ξ * z(x)) ** (-1 - 1 / ξ);
            }
        },
        ppf: function (u, ξ, miu, sigma) {
            if (ξ === 0) {
                return miu - sigma * Math.log(-Math.log(u));
            } else {
                return miu + (sigma * ((-Math.log(u)) ** -ξ - 1)) / ξ;
            }
        },
        sample: function (n_samples, ξ, miu, sigma) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), ξ, miu, sigma));
        },
        parametersRestrictions: function (ξ, miu, sigma) {
            const v0 = Boolean(1 + 0 * (ξ + miu));
            const v1 = sigma > 0;
            return v0 && v1;
        },
    },
    generalized_gamma: {
        cdf: function (x, a, d, p) {
            return jStat.lowRegGamma(d / p, (x / a) ** p);
        },
        pdf: function (x, a, d, p) {
            return ((p / a ** d) * x ** (d - 1) * Math.exp(-1 * (x / a) ** p)) / jStat.gammafn(d / p);
        },
        ppf: function (u, a, d, p) {
            return a * jStat.gammapinv(u, d / p) ** (1 / p);
        },
        sample: function (n_samples, a, d, p) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), a, d, p));
        },
        parametersRestrictions: function (a, d, p) {
            const v1 = a > 0;
            const v2 = d > 0;
            const v3 = p > 0;
            return v1 && v2 && v3;
        },
    },
    generalized_gamma_4p: {
        cdf: function (x, a, d, p, loc) {
            return jStat.lowRegGamma(d / p, ((x - loc) / a) ** p);
        },
        pdf: function (x, a, d, p, loc) {
            return ((p / a ** d) * (x - loc) ** (d - 1) * Math.exp(-1 * ((x - loc) / a) ** p)) / jStat.gammafn(d / p);
        },
        ppf: function (u, a, d, p, loc) {
            return a * jStat.gammapinv(u, d / p) ** (1 / p) + loc;
        },
        sample: function (n_samples, a, d, p, loc) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), a, d, p, loc));
        },
        parametersRestrictions: function (a, d, p, loc) {
            const v0 = Boolean(1 + 0 * loc);
            const v1 = a > 0;
            const v2 = d > 0;
            const v3 = p > 0;
            return v0 && v1 && v2 && v3;
        },
    },
    generalized_logistic: {
        cdf: function (x, c, loc, scale) {
            const z = (t: number): number => (t - loc) / scale;
            return 1 / (1 + Math.exp(-z(x))) ** c;
        },
        pdf: function (x, c, loc, scale) {
            const z = (t: number): number => (t - loc) / scale;
            return (c / scale) * Math.exp(-z(x)) * (1 + Math.exp(-z(x))) ** (-c - 1);
        },
        ppf: function (u, c, loc, scale) {
            return loc + scale * -Math.log(u ** (-1 / c) - 1);
        },
        sample: function (n_samples, c, loc, scale) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), c, loc, scale));
        },
        parametersRestrictions: function (c, loc, scale) {
            const v0 = Boolean(1 + 0 * loc);
            const v1 = scale > 0;
            const v2 = c > 0;
            return v0 && v1 && v2;
        },
    },
    generalized_normal: {
        cdf: function (x, miu, alpha, beta) {
            return 0.5 + (Math.sign(x - miu) / 2) * jStat.lowRegGamma(1 / beta, Math.abs((x - miu) / alpha) ** beta);
        },
        pdf: function (x, miu, alpha, beta) {
            return (beta / (2 * alpha * jStat.gammafn(1 / beta))) * Math.exp(-1 * (Math.abs(x - miu) / alpha) ** beta);
        },
        ppf: function (u, miu, alpha, beta) {
            return miu + Math.sign(u - 0.5) * (alpha ** beta * jStat.gammapinv(2 * Math.abs(u - 0.5), 1 / beta)) ** (1 / beta);
        },
        sample: function (n_samples, miu, alpha, beta) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), miu, alpha, beta));
        },
        parametersRestrictions: function (miu, alpha, beta) {
            const v0 = Boolean(1 + 0 * miu);
            const v1 = alpha > 0;
            const v2 = beta > 0;
            return v0 && v1 && v2;
        },
    },
    generalized_pareto: {
        cdf: function (x, c, miu, sigma) {
            const z = (t: number): number => (t - miu) / sigma;
            return 1 - (1 + c * z(x)) ** (-1 / c);
        },
        pdf: function (x, c, miu, sigma) {
            const z = (t: number): number => (t - miu) / sigma;
            return (1 / sigma) * (1 + c * z(x)) ** (-1 / c - 1);
        },
        ppf: function (u, c, miu, sigma) {
            return miu + (sigma * ((1 - u) ** -c - 1)) / c;
        },
        sample: function (n_samples, c, miu, sigma) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), c, miu, sigma));
        },
        parametersRestrictions: function (c, miu, sigma) {
            const v0 = Boolean(1 + 0 * miu);
            const v00 = Boolean(1 + 0 * c);
            const v1 = sigma > 0;
            return v0 && v00 && v1;
        },
    },
    gibrat: {
        cdf: function (x, loc, scale) {
            const z = (t: number): number => (t - loc) / scale;
            return 0.5 * (1 + jStat.erf(Math.log(z(x)) / Math.sqrt(2)));
        },
        pdf: function (x, loc, scale) {
            const z = (t: number): number => (t - loc) / scale;
            return (1 / (scale * z(x) * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.log(z(x)) ** 2);
        },
        ppf: function (u, loc, scale) {
            const inv_std_cdf = (t: number): number => Math.sqrt(2) * jStat.erfinv(2 * t - 1);
            return Math.exp(inv_std_cdf(u)) * scale + loc;
        },
        sample: function (n_samples, loc, scale) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), loc, scale));
        },
        parametersRestrictions: function (loc, scale) {
            const v0 = Boolean(1 + 0 * loc);
            const v1 = scale > 0;
            return v0 && v1;
        },
    },
    gumbel_left: {
        cdf: function (x, miu, sigma) {
            const z = (t: number): number => (t - miu) / sigma;
            return 1 - Math.exp(-Math.exp(z(x)));
        },
        pdf: function (x, miu, sigma) {
            const z = (t: number): number => (t - miu) / sigma;
            return (1 / sigma) * Math.exp(z(x) - Math.exp(z(x)));
        },
        ppf: function (u, miu, sigma) {
            return miu + sigma * Math.log(-Math.log(1 - u));
        },
        sample: function (n_samples, miu, sigma) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), miu, sigma));
        },
        parametersRestrictions: function (miu, sigma) {
            const v0 = Boolean(1 + 0 * miu);
            const v1 = sigma > 0;
            return v0 && v1;
        },
    },
    gumbel_right: {
        cdf: function (x, miu, sigma) {
            const z = (t: number): number => (t - miu) / sigma;
            return Math.exp(-Math.exp(-z(x)));
        },
        pdf: function (x, miu, sigma) {
            const z = (t: number): number => (t - miu) / sigma;
            return (1 / sigma) * Math.exp(-z(x) - Math.exp(-z(x)));
        },
        ppf: function (u, miu, sigma) {
            return miu - sigma * Math.log(-Math.log(u));
        },
        sample: function (n_samples, miu, sigma) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), miu, sigma));
        },
        parametersRestrictions: function (miu, sigma) {
            const v0 = Boolean(1 + 0 * miu);
            const v1 = sigma > 0;
            return v0 && v1;
        },
    },
    half_normal: {
        cdf: function (x, miu, sigma) {
            const z = (t: number): number => (t - miu) / sigma;
            return jStat.erf(z(x) / Math.sqrt(2));
        },
        pdf: function (x, miu, sigma) {
            const z = (t: number): number => (t - miu) / sigma;
            return (1 / sigma) * Math.sqrt(2 / Math.PI) * Math.exp(-(z(x) ** 2) / 2);
        },
        ppf: function (u, miu, sigma) {
            const inv_std_cdf = (t: number): number => Math.sqrt(2) * jStat.erfinv(2 * t - 1);
            return inv_std_cdf((1 + u) / 2) * sigma + miu;
        },
        sample: function (n_samples, miu, sigma) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), miu, sigma));
        },
        parametersRestrictions: function (miu, sigma) {
            const v0 = Boolean(1 + 0 * miu);
            const v1 = sigma > 0;
            return v0 && v1;
        },
    },
    hyperbolic_secant: {
        cdf: function (x, miu, sigma) {
            const z = (t: number): number => (Math.PI * (t - miu)) / (2 * sigma);
            return (2 / Math.PI) * Math.atan(Math.exp(z(x)));
        },
        pdf: function (x, miu, sigma) {
            const z = (t: number): number => (Math.PI * (t - miu)) / (2 * sigma);
            return 1 / Math.cosh(z(x)) / (2 * sigma);
        },
        ppf: function (u, miu, sigma) {
            return Math.log(Math.tan((u * Math.PI) / 2)) * ((2 * sigma) / Math.PI) + miu;
        },
        sample: function (n_samples, miu, sigma) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), miu, sigma));
        },
        parametersRestrictions: function (miu, sigma) {
            const v0 = Boolean(1 + 0 * miu);
            const v1 = sigma > 0;
            return v0 && v1;
        },
    },
    inverse_gamma: {
        cdf: function (x, alpha, beta) {
            return 1 - jStat.lowRegGamma(alpha, beta / x);
        },
        pdf: function (x, alpha, beta) {
            return (beta ** alpha * x ** (-alpha - 1) * Math.exp(-(beta / x))) / jStat.gammafn(alpha);
        },
        ppf: function (u, alpha, beta) {
            return beta / jStat.gammapinv(1 - u, alpha);
        },
        sample: function (n_samples, alpha, beta) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), alpha, beta));
        },
        parametersRestrictions: function (alpha, beta) {
            const v1 = alpha > 0;
            const v2 = beta > 0;
            return v1 && v2;
        },
    },
    inverse_gamma_3p: {
        cdf: function (x, alpha, beta, loc) {
            return 1 - jStat.lowRegGamma(alpha, beta / (x - loc));
        },
        pdf: function (x, alpha, beta, loc) {
            return (beta ** alpha * (x - loc) ** (-alpha - 1) * Math.exp(-(beta / (x - loc)))) / jStat.gammafn(alpha);
        },
        ppf: function (u, alpha, beta, loc) {
            return loc + beta / jStat.gammapinv(1 - u, alpha);
        },
        sample: function (n_samples, alpha, beta, loc) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), alpha, beta, loc));
        },
        parametersRestrictions: function (alpha, beta, loc) {
            const v0 = Boolean(1 + 0 * loc);
            const v1 = alpha > 0;
            const v2 = beta > 0;
            return v0 && v1 && v2;
        },
    },
    inverse_gaussian: {
        cdf: function (x, miu, lambda) {
            const std_cdf = (t: number): number => 0.5 * (1 + jStat.erf(t / Math.sqrt(2)));
            return std_cdf(Math.sqrt(lambda / x) * (x / miu - 1)) + Math.exp((2 * lambda) / miu) * std_cdf(-Math.sqrt(lambda / x) * (x / miu + 1));
        },
        pdf: function (x, miu, lambda) {
            return Math.sqrt(lambda / (2 * Math.PI * x ** 3)) * Math.exp(-(lambda * (x - miu) ** 2) / (2 * miu ** 2 * x));
        },
        ppf: function (u, miu, lambda) {
            const pyodideStore = usePyodideStore();
            const result = pyodideStore.runPython(`
                str(scipy.stats.invgauss.ppf(${u}, ${miu} / ${lambda}, scale=${u}))
            `);
            return Number(result);
        },
        sample: function (n_samples, miu, lambda) {
            const inv_std_cdf = (t: number): number => {
                return Math.sqrt(2) * jStat.erfinv(2 * t - 1);
            };
            return [...Array(n_samples).keys()].map(function () {
                const v = inv_std_cdf(Math.random());
                const y = v * v;
                const x = miu + (miu * miu * y) / (2 * lambda) - (miu / (2 * lambda)) * Math.sqrt(4 * miu * lambda * y + miu * miu * y * y);
                const u = Math.random();
                if (u <= miu / (miu + x)) {
                    return x;
                } else {
                    return (miu * miu) / x;
                }
            });
        },
        parametersRestrictions: function (miu, lambda) {
            const v1 = miu > 0;
            const v2 = lambda > 0;
            return v1 && v2;
        },
    },
    inverse_gaussian_3p: {
        cdf: function (x, miu, lambda, loc) {
            const std_cdf = (t: number): number => 0.5 * (1 + jStat.erf(t / Math.sqrt(2)));
            return std_cdf(Math.sqrt(lambda / (x - loc)) * ((x - loc) / miu - 1)) + Math.exp((2 * lambda) / miu) * std_cdf(-Math.sqrt(lambda / (x - loc)) * ((x - loc) / miu + 1));
        },
        pdf: function (x, miu, lambda, loc) {
            return Math.sqrt(lambda / (2 * Math.PI * (x - loc) ** 3)) * Math.exp(-(lambda * (x - loc - miu) ** 2) / (2 * miu ** 2 * (x - loc)));
        },
        ppf: function (u, miu, lambda, loc) {
            const pyodideStore = usePyodideStore();
            const result = pyodideStore.runPython(`
                str(scipy.stats.invgauss.ppf(${u}, ${miu} / ${lambda}, loc=${loc}, scale=${u}))
            `);
            return Number(result);
        },
        sample: function (n_samples, miu, lambda, loc) {
            const inv_std_cdf = (t: number): number => {
                return Math.sqrt(2) * jStat.erfinv(2 * t - 1);
            };
            return [...Array(n_samples).keys()].map(function () {
                const v = inv_std_cdf(Math.random());
                const y = v * v;
                const x = miu + (miu * miu * y) / (2 * lambda) - (miu / (2 * lambda)) * Math.sqrt(4 * miu * lambda * y + miu * miu * y * y);
                const u = Math.random();
                if (u <= miu / (miu + x)) {
                    return loc + x;
                } else {
                    return loc + (miu * miu) / x;
                }
            });
        },
        parametersRestrictions: function (miu, lambda, loc) {
            const v0 = Boolean(1 + 0 * loc);
            const v1 = miu > 0;
            const v2 = lambda > 0;
            return v0 && v1 && v2;
        },
    },
    johnson_sb: {
        cdf: function (x, xi, lambda, gamma, delta) {
            const std_cdf = (t: number): number => 0.5 * (1 + jStat.erf(t / Math.sqrt(2)));
            const z = (t: number): number => (t - xi) / lambda;
            return std_cdf(gamma + delta * Math.log(z(x) / (1 - z(x))));
        },
        pdf: function (x, xi, lambda, gamma, delta) {
            const z = (t: number): number => (t - xi) / lambda;
            return (delta / (lambda * Math.sqrt(2 * Math.PI) * z(x) * (1 - z(x)))) * Math.exp(-(1 / 2) * (gamma + delta * Math.log(z(x) / (1 - z(x)))) ** 2);
        },
        ppf: function (u, xi, lambda, gamma, delta) {
            const inv_std_cdf = (t: number): number => Math.sqrt(2) * jStat.erfinv(2 * t - 1);
            return (lambda * Math.exp((inv_std_cdf(u) - gamma) / delta)) / (1 + Math.exp((inv_std_cdf(u) - gamma) / delta)) + xi;
        },
        sample: function (n_samples, xi, lambda, gamma, delta) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), xi, lambda, gamma, delta));
        },
        parametersRestrictions: function (xi, lambda, gamma, delta) {
            const v0 = Boolean(1 + 0 * (xi + gamma));
            const v1 = delta > 0;
            const v2 = lambda > 0;
            return v0 && v1 && v2;
        },
    },
    johnson_su: {
        cdf: function (x, xi, lambda, gamma, delta) {
            const std_cdf = (t: number): number => 0.5 * (1 + jStat.erf(t / Math.sqrt(2)));
            const z = (t: number): number => (t - xi) / lambda;
            return std_cdf(gamma + delta * Math.asinh(z(x)));
        },
        pdf: function (x, xi, lambda, gamma, delta) {
            const z = (t: number): number => (t - xi) / lambda;
            return (delta / (lambda * Math.sqrt(2 * Math.PI) * Math.sqrt(z(x) ** 2 + 1))) * Math.exp(-(1 / 2) * (gamma + delta * Math.asinh(z(x))) ** 2);
        },
        ppf: function (u, xi, lambda, gamma, delta) {
            const inv_std_cdf = (t: number): number => Math.sqrt(2) * jStat.erfinv(2 * t - 1);
            return lambda * Math.sinh((inv_std_cdf(u) - gamma) / delta) + xi;
        },
        sample: function (n_samples, xi, lambda, gamma, delta) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), xi, lambda, gamma, delta));
        },
        parametersRestrictions: function (xi, lambda, gamma, delta) {
            const v0 = Boolean(1 + 0 * (xi + gamma));
            const v1 = delta > 0;
            const v2 = lambda > 0;
            return v0 && v1 && v2;
        },
    },
    kumaraswamy: {
        cdf: function (x, alpha, beta, min, max) {
            const z = (t: number): number => (t - min) / (max - min);
            return 1 - (1 - z(x) ** alpha) ** beta;
        },
        pdf: function (x, alpha, beta, min, max) {
            const z = (t: number): number => (t - min) / (max - min);
            return (alpha * beta * z(x) ** (alpha - 1) * (1 - z(x) ** alpha) ** (beta - 1)) / (max - min);
        },
        ppf: function (u, alpha, beta, min, max) {
            return (1 - (1 - u) ** (1 / beta)) ** (1 / alpha) * (max - min) + min;
        },
        sample: function (n_samples, alpha, beta, min, max) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), alpha, beta, min, max));
        },
        parametersRestrictions: function (alpha, beta, min, max) {
            const v1 = alpha > 0;
            const v2 = beta > 0;
            const v3 = min < max;
            return v1 && v2 && v3;
        },
    },
    laplace: {
        cdf: function (x, miu, b) {
            return 0.5 + 0.5 * Math.sign(x - miu) * (1 - Math.exp(-Math.abs(x - miu) / b));
        },
        pdf: function (x, miu, b) {
            return (1 / (2 * b)) * Math.exp(-Math.abs(x - miu) / b);
        },
        ppf: function (u, miu, b) {
            return miu - b * Math.sign(u - 0.5) * Math.log(1 - 2 * Math.abs(u - 0.5));
        },
        sample: function (n_samples, miu, b) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), miu, b));
        },
        parametersRestrictions: function (miu, b) {
            const v0 = Boolean(1 + 0 * miu);
            const v1 = b > 0;
            return v0 && v1;
        },
    },
    levy: {
        cdf: function (x, miu, c) {
            const y = (t: number): number => Math.sqrt(c / (t - miu));
            return jStat.erfc(y(x) / Math.sqrt(2));
        },
        pdf: function (x, miu, c) {
            return (Math.sqrt(c / (2 * Math.PI)) * Math.exp(-c / (2 * (x - miu)))) / (x - miu) ** 1.5;
        },
        ppf: function (u, miu, c) {
            const inv_std_cdf = (t: number): number => Math.sqrt(2) * jStat.erfinv(2 * t - 1);
            return miu + c / inv_std_cdf((2 - u) / 2) ** 2;
        },
        sample: function (n_samples, miu, c) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), miu, c));
        },
        parametersRestrictions: function (miu, c) {
            const v0 = Boolean(1 + 0 * miu);
            const v1 = c > 0;
            return v0 && v1;
        },
    },
    loggamma: {
        cdf: function (x, c, miu, sigma) {
            const z = (x: number): number => (x - miu) / sigma;
            return jStat.lowRegGamma(c, Math.exp(z(x)));
        },
        pdf: function (x, c, miu, sigma) {
            const z = (x: number): number => (x - miu) / sigma;
            return Math.exp(c * z(x) - Math.exp(z(x)) - jStat.gammaln(c)) / sigma;
        },
        ppf: function (u, c, miu, sigma) {
            return miu + sigma * Math.log(jStat.gammapinv(u, c));
        },
        sample: function (n_samples, c, miu, sigma) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), c, miu, sigma));
        },
        parametersRestrictions: function (c, miu, sigma) {
            const v0 = Boolean(1 + 0 * miu);
            const v1 = c > 0;
            const v2 = sigma > 0;
            return v0 && v1 && v2;
        },
    },
    logistic: {
        cdf: function (x, miu, sigma) {
            const z = (t: number): number => Math.exp(-(t - miu) / sigma);
            return 1 / (1 + z(x));
        },
        pdf: function (x, miu, sigma) {
            const z = (t: number): number => Math.exp(-(t - miu) / sigma);
            return z(x) / (sigma * (1 + z(x)) ** 2);
        },
        ppf: function (u, miu, sigma) {
            return miu + sigma * Math.log(u / (1 - u));
        },
        sample: function (n_samples, miu, sigma) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), miu, sigma));
        },
        parametersRestrictions: function (miu, sigma) {
            const v0 = Boolean(1 + 0 * miu);
            const v1 = sigma > 0;
            return v0 && v1;
        },
    },
    loglogistic: {
        cdf: function (x, alpha, beta) {
            return x ** beta / (alpha ** beta + x ** beta);
        },
        pdf: function (x, alpha, beta) {
            return ((beta / alpha) * (x / alpha) ** (beta - 1)) / (1 + (x / alpha) ** beta) ** 2;
        },
        ppf: function (u, alpha, beta) {
            return alpha * (u / (1 - u)) ** (1 / beta);
        },
        sample: function (n_samples, alpha, beta) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), alpha, beta));
        },
        parametersRestrictions: function (alpha, beta) {
            const v1 = alpha > 0;
            const v2 = beta > 0;
            return v1 && v2;
        },
    },
    loglogistic_3p: {
        cdf: function (x, alpha, beta, loc) {
            return (x - loc) ** beta / (alpha ** beta + (x - loc) ** beta);
        },
        pdf: function (x, alpha, beta, loc) {
            return ((beta / alpha) * ((x - loc) / alpha) ** (beta - 1)) / (1 + ((x - loc) / alpha) ** beta) ** 2;
        },
        ppf: function (u, alpha, beta, loc) {
            return alpha * (u / (1 - u)) ** (1 / beta) + loc;
        },
        sample: function (n_samples, alpha, beta, loc) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), alpha, beta, loc));
        },
        parametersRestrictions: function (alpha, beta, loc) {
            const v0 = Boolean(1 + 0 * loc);
            const v1 = alpha > 0;
            const v2 = beta > 0;
            return v0 && v1 && v2;
        },
    },
    lognormal: {
        cdf: function (x, miu, sigma) {
            const std_cdf = (t: number): number => 0.5 * (1 + jStat.erf(t / Math.sqrt(2)));
            return std_cdf((Math.log(x) - miu) / sigma);
        },
        pdf: function (x, miu, sigma) {
            return (1 / (x * sigma * Math.sqrt(2 * Math.PI))) * Math.exp(-((Math.log(x) - miu) ** 2 / (2 * sigma ** 2)));
        },
        ppf: function (u, miu, sigma) {
            const inv_std_cdf = (t: number): number => Math.sqrt(2) * jStat.erfinv(2 * t - 1);
            return Math.exp(miu + sigma * inv_std_cdf(u));
        },
        sample: function (n_samples, miu, sigma) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), miu, sigma));
        },
        parametersRestrictions: function (miu, sigma) {
            const v1 = miu > 0;
            const v2 = sigma > 0;
            return v1 && v2;
        },
    },
    maxwell: {
        cdf: function (x, alpha, loc) {
            const z = (t: number): number => (t - loc) / alpha;
            return jStat.erf(z(x) / Math.sqrt(2)) - Math.sqrt(2 / Math.PI) * z(x) * Math.exp((-1 * z(x) ** 2) / 2);
        },
        pdf: function (x, alpha, loc) {
            const z = (t: number): number => (t - loc) / alpha;
            return (1 / alpha) * Math.sqrt(2 / Math.PI) * z(x) ** 2 * Math.exp((-1 * z(x) ** 2) / 2);
        },
        ppf: function (u, alpha, loc) {
            return alpha * Math.sqrt(2 * jStat.gammapinv(u, 1.5)) + loc;
        },
        sample: function (n_samples, alpha, loc) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), alpha, loc));
        },
        parametersRestrictions: function (alpha, loc) {
            const v0 = Boolean(1 + 0 * loc);
            const v1 = alpha > 0;
            return v0 && v1;
        },
    },
    moyal: {
        cdf: function (x, miu, sigma) {
            const z = (t: number): number => (t - miu) / sigma;
            return jStat.erfc(Math.exp(-0.5 * z(x)) / Math.sqrt(2));
        },
        pdf: function (x, miu, sigma) {
            const z = (t: number): number => (t - miu) / sigma;
            return Math.exp(-0.5 * (z(x) + Math.exp(-z(x)))) / (sigma * Math.sqrt(2 * Math.PI));
        },
        ppf: function (u, miu, sigma) {
            const inv_std_cdf = (t: number): number => Math.sqrt(2) * jStat.erfinv(2 * t - 1);
            return miu - sigma * Math.log(inv_std_cdf(1 - u / 2) ** 2);
        },
        sample: function (n_samples, miu, sigma) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), miu, sigma));
        },
        parametersRestrictions: function (miu, sigma) {
            const v0 = Boolean(1 + 0 * miu);
            const v1 = sigma > 0;
            return v0 && v1;
        },
    },
    nakagami: {
        cdf: function (x, m, omega) {
            return jStat.lowRegGamma(m, (m / omega) * x ** 2);
        },
        pdf: function (x, m, omega) {
            return ((2 * m ** m) / (jStat.gammafn(m) * omega ** m)) * (x ** (2 * m - 1) * Math.exp(-(m / omega) * x ** 2));
        },
        ppf: function (u, m, omega) {
            return Math.sqrt(jStat.gammapinv(u, m) * (omega / m));
        },
        sample: function (n_samples, m, omega) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), m, omega));
        },
        parametersRestrictions: function (m, omega) {
            const v1 = m >= 0.5;
            const v2 = omega > 0;
            return v1 && v2;
        },
    },
    nc_chi_square: {
        cdf: function (x, lambda, n) {
            const pyodideStore = usePyodideStore();
            const result = pyodideStore.runPython(`
                str(scipy.stats.ncx2.cdf(${x}, ${lambda}, ${n}))
            `);
            return Number(result);
        },
        pdf: function (x, lambda, n) {
            return (1 / 2) * Math.exp(-(x + lambda) / 2) * (x / lambda) ** ((n - 2) / 4) * BESSEL.besseli(Math.sqrt(lambda * x), (n - 2) / 2);
        },
        ppf: function (u, lambda, n) {
            const pyodideStore = usePyodideStore();
            const result = pyodideStore.runPython(`
                str(scipy.stats.ncx2.ppf(${u}, ${lambda}, ${n}))
            `);
            return Number(result);
        },
        sample: function (n_samples, lambda, n) {
            // This method return a value from sample.
            // But not the specific inverse of (u)
            const norm_inv = (t: number, miu_: number, sigma_: number): number => {
                return miu_ + sigma_ * Math.sqrt(2) * jStat.erfinv(2 * t - 1);
            };
            return [...Array(n_samples).keys()].map(function () {
                const result = [...Array(n).keys()].reduce((acum) => acum + norm_inv(Math.random(), Math.sqrt(lambda / n), 1) ** 2, 0);
                return result;
            });
        },
        parametersRestrictions: function (lambda, n) {
            const v1 = lambda > 0;
            const v2 = n > 0;
            return v1 && v2;
        },
    },
    nc_f: {
        cdf: function (x, lambda, n1, n2) {
            const pyodideStore = usePyodideStore();
            const result = pyodideStore.runPython(`
                str(scipy.stats.ncf.cdf(${x}, ${n1}, ${n2}, ${lambda}))
            `);
            return Number(result);
        },
        pdf: function (x, lambda, n1, n2) {
            const pyodideStore = usePyodideStore();
            const result = pyodideStore.runPython(`
                str(scipy.stats.ncf.pdf(${x}, ${n1}, ${n2}, ${lambda}))
            `);
            return Number(result);
        },
        ppf: function (u, lambda, n1, n2) {
            const pyodideStore = usePyodideStore();
            const result = pyodideStore.runPython(`
                str(scipy.stats.ncf.ppf(${u}, ${n1}, ${n2}, ${lambda}))
            `);
            return Number(result);
        },
        sample: function (n_samples, lambda, n1, n2) {
            // This method return a value from sample.
            // But not the specific inverse of (u)
            const norm_inv = (t: number, miu_: number, sigma_: number): number => {
                return miu_ + sigma_ * Math.sqrt(2) * jStat.erfinv(2 * t - 1);
            };
            return [...Array(n_samples).keys()].map(function () {
                const var_nc_chi2 = [...Array(n1).keys()].reduce((acum) => acum + norm_inv(Math.random(), Math.sqrt(lambda / n1), 1) ** 2, 0);
                const var_chi2 = 2 * jStat.gammapinv(Math.random(), n2 / 2);
                return var_nc_chi2 / n1 / (var_chi2 / n2);
            });
        },
        parametersRestrictions: function (lambda, n1, n2) {
            const v1 = lambda > 0;
            const v2 = n1 > 0;
            const v3 = n2 > 0;
            return v1 && v2 && v3;
        },
    },
    nc_t_student: {
        cdf: function (x, lambda, n, loc, scale) {
            const z = (t: number): number => (t - loc) / scale;
            const pyodideStore = usePyodideStore();
            const result = pyodideStore.runPython(`
                scipy.stats.nct.cdf(${z(x)}, ${n}, ${lambda})
            `);
            return Number(result);
        },
        pdf: function (x, lambda, n, loc, scale) {
            const z = (t: number): number => (t - loc) / scale;
            const pyodideStore = usePyodideStore();
            const result = pyodideStore.runPython(`
                scipy.stats.nct.pdf(${z(x)}, ${n}, ${lambda}) / ${scale}
            `);
            return Number(result);
        },
        ppf: function (u, lambda, n, loc, scale) {
            const pyodideStore = usePyodideStore();
            const result = pyodideStore.runPython(`
                scipy.stats.nct.ppf(${u}, ${n}, ${lambda}, loc = ${loc}, scale = ${scale})
            `);
            return Number(result);
        },
        sample: function (n_samples, lambda, n, loc, scale) {
            // This method return a value from sample.
            // But not the specific inverse of (u)
            const norm_inv = (t: number, miu_: number, sigma_: number): number => {
                return miu_ + sigma_ * Math.sqrt(2) * jStat.erfinv(2 * t - 1);
            };
            return [...Array(n_samples).keys()].map(function () {
                const var_normal = norm_inv(Math.random(), lambda, 1);
                const var_chi2 = 2 * jStat.gammapinv(Math.random(), n / 2);
                return loc + (scale * var_normal) / Math.sqrt(var_chi2 / n);
            });
        },
        parametersRestrictions: function (lambda, n, loc, scale) {
            const v0 = Boolean(1 + 0 * lambda);
            const v1 = Boolean(1 + 0 * loc);
            const v2 = scale > 0;
            const v3 = n > 0;
            return v0 && v1 && v2 && v3;
        },
    },
    normal: {
        cdf: function (x, miu, sigma) {
            const std_cdf = (t: number): number => 0.5 * (1 + jStat.erf(t / Math.sqrt(2)));
            return std_cdf((x - miu) / sigma);
        },
        pdf: function (x, miu, sigma) {
            return (1 / (sigma * Math.sqrt(2 * Math.PI))) * Math.exp(-((x - miu) ** 2 / (2 * sigma ** 2)));
        },
        ppf: function (u, miu, sigma) {
            const inv_std_cdf = (t: number): number => Math.sqrt(2) * jStat.erfinv(2 * t - 1);
            return miu + sigma * inv_std_cdf(u);
        },
        sample: function (n_samples, miu, sigma) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), miu, sigma));
        },
        parametersRestrictions: function (miu, sigma) {
            const v0 = Boolean(1 + 0 * miu);
            const v1 = sigma > 0;
            return v0 && v1;
        },
    },
    pareto_first_kind: {
        cdf: function (x, xm, alpha, loc) {
            return 1 - (xm / (x - loc)) ** alpha;
        },
        pdf: function (x, xm, alpha, loc) {
            return (alpha * xm ** alpha) / (x - loc) ** (alpha + 1);
        },
        ppf: function (u, xm, alpha, loc) {
            return loc + xm * ((1 - u) ** -(1 / alpha));
        },
        sample: function (n_samples, xm, alpha, loc) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), xm, alpha, loc));
        },
        parametersRestrictions: function (xm, alpha, loc) {
            const v0 = Boolean(1 + 0 * loc);
            const v1 = xm > 0;
            const v2 = alpha > 0;
            return v0 && v1 && v2;
        },
    },
    pareto_second_kind: {
        cdf: function (x, xm, alpha, loc) {
            return 1 - (xm / (x - loc + xm)) ** alpha;
        },
        pdf: function (x, xm, alpha, loc) {
            return (alpha * xm ** alpha) / (x - loc + xm) ** (alpha + 1);
        },
        ppf: function (u, xm, alpha, loc) {
            return loc + xm / (1 - u) ** (1 / alpha) - xm;
        },
        sample: function (n_samples, xm, alpha, loc) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), xm, alpha, loc));
        },
        parametersRestrictions: function (xm, alpha, loc) {
            const v0 = Boolean(1 + 0 * loc);
            const v1 = xm > 0;
            const v2 = alpha > 0;
            return v0 && v1 && v2;
        },
    },
    pert: {
        cdf: function (x, a, b, c) {
            const z = (t: number, A: number, B: number): number => (t - A) / (B - A);
            const α1 = (4 * b + c - 5 * a) / (c - a);
            const α2 = (5 * c - a - 4 * b) / (c - a);
            return jStat.ibeta(z(x, a, c), α1, α2);
        },
        pdf: function (x, a, b, c) {
            const α1 = (4 * b + c - 5 * a) / (c - a);
            const α2 = (5 * c - a - 4 * b) / (c - a);
            return ((x - a) ** (α1 - 1) * (c - x) ** (α2 - 1)) / (jStat.betafn(α1, α2) * (c - a) ** (α1 + α2 - 1));
        },
        ppf: function (u, a, b, c) {
            const α1 = (4 * b + c - 5 * a) / (c - a);
            const α2 = (5 * c - a - 4 * b) / (c - a);
            return a + (c - a) * jStat.ibetainv(u, α1, α2);
        },
        sample: function (n_samples, a, b, c) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), a, b, c));
        },
        parametersRestrictions: function (a, b, c) {
            const v1 = a < b;
            const v2 = b < c;
            return v1 && v2;
        },
    },
    power_function: {
        cdf: function (x, alpha, a, b) {
            return ((x - a) / (b - a)) ** alpha;
        },
        pdf: function (x, alpha, a, b) {
            return (alpha * (x - a) ** (alpha - 1)) / (b - a) ** alpha;
        },
        ppf: function (u, alpha, a, b) {
            return u ** (1 / alpha) * (b - a) + a;
        },
        sample: function (n_samples, alpha, a, b) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), alpha, a, b));
        },
        parametersRestrictions: function (alpha, a, b) {
            const v1 = alpha > 0;
            const v2 = b > a;
            return v1 && v2;
        },
    },
    rayleigh: {
        cdf: function (x, gamma, sigma) {
            return 1 - Math.exp(-0.5 * ((x - gamma) / sigma) ** 2);
        },
        pdf: function (x, gamma, sigma) {
            return ((x - gamma) / sigma ** 2) * Math.exp(-0.5 * ((x - gamma) / sigma) ** 2);
        },
        ppf: function (u, gamma, sigma) {
            return Math.sqrt(-2 * Math.log(1 - u)) * sigma + gamma;
        },
        sample: function (n_samples) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random()));
        },
        parametersRestrictions: function (gamma, sigma) {
            const v0 = Boolean(1 + 0 * gamma);
            const v1 = sigma > 0;
            return v0 && v1;
        },
    },
    reciprocal: {
        cdf: function (x, a, b) {
            return (Math.log(x) - Math.log(a)) / (Math.log(b) - Math.log(a));
        },
        pdf: function (x, a, b) {
            return 1 / (x * (Math.log(b) - Math.log(a)));
        },
        ppf: function (u, a, b) {
            return Math.exp(u * (Math.log(b) - Math.log(a)) + Math.log(a));
        },
        sample: function (n_samples, a, b) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), a, b));
        },
        parametersRestrictions: function (a, b) {
            const v1 = b > a;
            return v1;
        },
    },
    rice: {
        cdf: function (x, v, sigma) {
            function Q(M: number, a: number, b: number): number {
                // Marcum Q-function
                // https://en.wikipedia.org/wiki/Marcum_Q-function
                let k = 1 - M;
                let x = (a / b) ** k * BESSEL.besseli(a * b, k);
                let acum = 0;
                while (x > 1e-20) {
                    acum += x;
                    k += 1;
                    x = (a / b) ** k * BESSEL.besseli(a * b, k);
                }
                const result = acum * Math.exp(-(a ** 2 + b ** 2) / 2);
                return result;
            }
            return 1 - Q(1, v / sigma, x / sigma);
        },
        pdf: function (x, v, sigma) {
            return (x / sigma ** 2) * Math.exp(-(x ** 2 + v ** 2) / (2 * sigma ** 2)) * BESSEL.besseli((x * v) / sigma ** 2, 0);
        },
        ppf: function (u, v, sigma) {
            const pyodideStore = usePyodideStore();
            const result = pyodideStore.runPython(`
                scipy.stats.rice.pdf(${u}, ${v}/${sigma}, scale=${sigma})
            `);
            return Number(result);
        },
        sample: function (n_samples, v, sigma) {
            // This method return a value from sample.
            // But not the specific inverse of (u)
            const norm_inv = (t: number, miu_: number, sigma_: number): number => {
                return miu_ + sigma_ * Math.sqrt(2) * jStat.erfinv(2 * t - 1);
            };
            return [...Array(n_samples).keys()].map(function () {
                const θ = Math.PI / 2;
                const r1 = Math.random();
                const r2 = Math.random();
                const X = norm_inv(r1, v * Math.cos(θ), sigma);
                const Y = norm_inv(r2, v * Math.sin(θ), sigma);
                return Math.sqrt(X * X + Y * Y);
            });
        },
        parametersRestrictions: function (v, sigma) {
            const v1 = v > 0;
            const v2 = sigma > 0;
            return v1 && v2;
        },
    },
    semicircular: {
        cdf: function (x, loc, R) {
            const z = (t: number): number => t - loc;
            return 0.5 + (z(x) * Math.sqrt(R ** 2 - z(x) ** 2)) / (Math.PI * R ** 2) + Math.asin(z(x) / R) / Math.PI;
        },
        pdf: function (x, loc, R) {
            const z = (t: number): number => t - loc;
            return (2 * Math.sqrt(R ** 2 - z(x) ** 2)) / (Math.PI * R ** 2);
        },
        ppf: function (u, loc, R) {
            return loc + R * (2 * jStat.ibetainv(u, 1.5, 1.5) - 1);
        },
        sample: function (n_samples, loc, R) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), loc, R));
        },
        parametersRestrictions: function (loc, R) {
            const v0 = Boolean(1 + 0 * loc);
            const v1 = R > 0;
            return v0 && v1;
        },
    },
    t_student: {
        cdf: function (x, df) {
            return jStat.ibeta((x + Math.sqrt(x * x + df)) / (2 * Math.sqrt(x * x + df)), df / 2, df / 2);
        },
        pdf: function (x, df) {
            return (1 + (x * x) / df) ** (-(df + 1) / 2) / (Math.sqrt(df) * jStat.betafn(0.5, df / 2));
        },
        ppf: function (u, df) {
            if (u >= 0.5) {
                return Math.sqrt((df * (1 - jStat.ibetainv(2 * Math.min(u, 1 - u), 0.5 * df, 0.5))) / jStat.ibetainv(2 * Math.min(u, 1 - u), 0.5 * df, 0.5));
            } else {
                return -Math.sqrt((df * (1 - jStat.ibetainv(2 * Math.min(u, 1 - u), 0.5 * df, 0.5))) / jStat.ibetainv(2 * Math.min(u, 1 - u), 0.5 * df, 0.5));
            }
        },
        sample: function (n_samples, df) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), df));
        },
        parametersRestrictions: function (df) {
            const v1 = df > 0;
            return v1;
        },
    },
    t_student_3p: {
        cdf: function (x, df, loc, scale) {
            const z = (t: number): number => (t - loc) / scale;
            return jStat.ibeta((z(x) + Math.sqrt(z(x) ** 2 + df)) / (2 * Math.sqrt(z(x) ** 2 + df)), df / 2, df / 2);
        },
        pdf: function (x, df, loc, scale) {
            const z = (t: number): number => (t - loc) / scale;
            return ((1 / (Math.sqrt(df) * jStat.betafn(0.5, df / 2))) * (1 + z(x) ** 2 / df) ** (-(df + 1) / 2)) / scale;
        },
        ppf: function (u, df, loc, scale) {
            if (u >= 0.5) {
                return loc + scale * Math.sqrt((df * (1 - jStat.ibetainv(2 * Math.min(u, 1 - u), 0.5 * df, 0.5))) / jStat.ibetainv(2 * Math.min(u, 1 - u), 0.5 * df, 0.5));
            } else {
                return loc - scale * Math.sqrt((df * (1 - jStat.ibetainv(2 * Math.min(u, 1 - u), 0.5 * df, 0.5))) / jStat.ibetainv(2 * Math.min(u, 1 - u), 0.5 * df, 0.5));
            }
        },
        sample: function (n_samples, df, loc, scale) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), df, loc, scale));
        },
        parametersRestrictions: function (df, loc, scale) {
            const v0 = Boolean(1 + 0 * loc);
            const v1 = df > 0;
            const v2 = scale > 0;
            return v0 && v1 && v2;
        },
    },
    trapezoidal: {
        cdf: function (x, a, b, c, d) {
            if (x <= a) {
                return 0;
            }
            if (a <= x && x < b) {
                return (1 / (d + c - b - a)) * (1 / (b - a)) * (x - a) ** 2;
            }
            if (b <= x && x < c) {
                return (1 / (d + c - b - a)) * (2 * x - a - b);
            }
            if (c <= x && x <= d) {
                return 1 - (1 / (d + c - b - a)) * (1 / (d - c)) * (d - x) ** 2;
            }
            if (x >= d) {
                return 1;
            }
            return 0;
        },
        pdf: function (x, a, b, c, d) {
            if (x <= a) {
                return 0;
            }
            if (a <= x && x < b) {
                return (2 / (d + c - b - a)) * ((x - a) / (b - a));
            }
            if (b <= x && x < c) {
                return 2 / (d + c - b - a);
            }
            if (c <= x && x <= d) {
                return (2 / (d + c - b - a)) * ((d - x) / (d - c));
            }
            if (x >= d) {
                return 0;
            }
            return 0;
        },
        ppf: function (u, a, b, c, d) {
            const h = 2 / (d + c - a - b);
            const Area1 = ((b - a) * h) / 2;
            const Area2 = (c - b) * h;
            const Area3 = ((d - c) * h) / 2;
            if (u <= Area1) {
                return a + Math.sqrt(u * (d + c - b - a) * (b - a));
            }
            if (u <= Area1 + Area2) {
                return (u * (d + c - b - a) + a + b) / 2;
            }
            if (u <= Area1 + Area2 + Area3) {
                return d - Math.sqrt((1 - u) * (d + c - b - a) * (d - c));
            }
            return 0;
        },
        sample: function (n_samples, a, b, c, d) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), a, b, c, d));
        },
        parametersRestrictions: function (a, b, c, d) {
            const v1 = a < b;
            const v2 = b < c;
            const v3 = c < d;
            return v1 && v2 && v3;
        },
    },
    triangular: {
        cdf: function (x, a, b, c) {
            if (x <= a) {
                return 0;
            }
            if (a < x && x <= c) {
                return (x - a) ** 2 / ((b - a) * (c - a));
            }
            if (c < x && x < b) {
                return 1 - (b - x) ** 2 / ((b - a) * (b - c));
            }
            if (x >= c) {
                return 1;
            }
            return 0;
        },
        pdf: function (x, a, b, c) {
            if (x <= a) {
                return 0;
            }
            if (a <= x && x <= c) {
                return (2 * (x - a)) / ((b - a) * (c - a));
            }
            if (x > c && x <= b) {
                return (2 * (b - x)) / ((b - a) * (b - c));
            }
            if (x >= c) {
                return 0;
            }
            return 0;
        },
        ppf: function (u, a, b, c) {
            if (u < (c - a) / (b - a)) {
                return a + Math.sqrt(u * (b - a) * (c - a));
            } else {
                return b - Math.sqrt((1 - u) * (b - a) * (b - c));
            }
        },
        sample: function (n_samples, a, b, c) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), a, b, c));
        },
        parametersRestrictions: function (a, b, c) {
            const v1 = a < c;
            const v2 = c < b;
            return v1 && v2;
        },
    },
    uniform: {
        cdf: function (x, a, b) {
            return (x - a) / (b - a);
        },
        pdf: function (x, a, b) {
            return 1 / (b - a);
        },
        ppf: function (u, a, b) {
            return a + u * (b - a);
        },
        sample: function (n_samples, a, b) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), a, b));
        },
        parametersRestrictions: function (a, b) {
            const v1 = b > a;
            return v1;
        },
    },
    weibull: {
        cdf: function (x, alpha, beta) {
            return 1 - Math.exp(-1 * (x / beta) ** alpha);
        },
        pdf: function (x, alpha, beta) {
            return (alpha / beta) * (x / beta) ** (alpha - 1) * Math.exp(-1 * (x / beta) ** alpha);
        },
        ppf: function (u, alpha, beta) {
            return beta * (-Math.log(1 - u)) ** (1 / alpha);
        },
        sample: function (n_samples, alpha, beta) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), alpha, beta));
        },
        parametersRestrictions: function (alpha, beta) {
            const v1 = alpha > 0;
            const v2 = beta > 0;
            return v1 && v2;
        },
    },
    weibull_3p: {
        cdf: function (x, alpha, beta, loc) {
            const z = (t: number): number => (t - loc) / beta;
            return 1 - Math.exp(-1 * z(x) ** alpha);
        },
        pdf: function (x, alpha, beta, loc) {
            const z = (t: number): number => (t - loc) / beta;
            return (alpha / beta) * z(x) ** (alpha - 1) * Math.exp(-1 * z(x) ** alpha);
        },
        ppf: function (u, alpha, beta, loc) {
            return loc + beta * (-Math.log(1 - u)) ** (1 / alpha);
        },
        sample: function (n_samples, alpha, beta, loc) {
            return [...Array(n_samples).keys()].map(() => this.ppf(Math.random(), alpha, beta, loc));
        },
        parametersRestrictions: function (alpha, beta, loc) {
            const v0 = Boolean(1 + 0 * loc);
            const v1 = alpha > 0;
            const v2 = beta > 0;
            return v0 && v1 && v2;
        },
    },
};

export { continuousDistributions };