const eqs = `
BERNOULLI
X\\sim\\mathrm{Bernoulli}\\left(p\\right)\\\\
x\\in\\left\\{0,1\\right\\}\\\\
p\\in\\left(0,1\\right)\\subseteq\\mathbb{R}\\\\
F_{X}\\left(x\\right)=\\left\\{\\begin{array}{cl} 1-p & \\text{if } \\ x=0 \\\\ 1 & \\text{if } \\ x=1 \\end{array} \\right.\\\\
f_{X}\\left(x\\right)=p^x(1-p)^{1-x}\\\\
F^{-1}_{X}\\left(u\\right)=\\left\\{\\begin{array}{cl} 1 & \\text{if } \\ u \\leq p \\\\ 0 & \\text{if } \\ u \\gt p \\end{array} \\right.\\\\
E[X^k]=\\mu'_{k}=\\sum_{x=0}^{1}x^{k}f_{X}\\left(x\\right)=p\\\\
\\mathrm{Mean}(X)=\\mu'_{1}=p\\\\
\\mathrm{Variance}(X)=(\\mu'_{2}-\\mu'^{2}_{1})=p(1-p)\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}=\\frac{1-2p}{\\sqrt{p(1-p)}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}=3+\\frac{1 - 6p(1-p)}{p(1-p)}\\\\
\\mathrm{Median}(X)=\\left\\{\\begin{array}{cl} 0 & \\text{if } p < 1/2\\\\ \\left[0, 1\\right] & \\text{if } p = 1/2\\\\ 1 & \\text{if } p > 1/2 \\end{array} \\right.\\\\
\\mathrm{Mode}(X)=\\left\\{\\begin{array}{cl} 0 & \\text{if } \\ p < 1/2\\\\ 0, 1 & \\text{if } \\ p = 1/2\\\\ 1 & \\text{if } \\ p > 1/2 \\end{array} \\right.\\\\
u:\\text{Uniform[0,1] random varible}\\\\

BINOMIAL
X\\sim\\mathrm{Binomial}\\left(n,p\\right)\\\\
x\\in\\mathbb{N}\\equiv \\left\\{ 0,1,2,\\dots \\right\\}\\\\
n\\in\\mathbb{N};p\\in\\left(0,1\\right)\\subseteq\\mathbb{R}\\\\
F_{X}\\left(x\\right)=\\sum_{i=0}^{x} {n\\choose i}p^i(1-p)^{n-i}=I(1-p, n - x, 1 + x)\\\\
f_{X}\\left(x\\right)=\\binom{n}{x} p^x (1-p)^{n-x}\\\\
F^{-1}_{X}\\left(u\\right)=\\arg\\min_{x}\\left| F_{X}\\left(x\\right)-u \\right|\\\\
E[X^k]=\\mu'_{k}=\\sum_{x=0}^{\\infty }x^{k}f_{X}\\left(x\\right)=\\sum_{i=0}^k\\tfrac{n!}{(n-i)!}S(k,i)p^{i}\\\\
\\mathrm{Mean}(X)=\\mu'_{1}=np\\\\
\\mathrm{Variance}(X)=(\\mu'_{2}-\\mu'^{2}_{1})=np(1-p)\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}=\\frac{1-2p}{\\sqrt{np(1-p)}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}=3+\\frac{1-6p(1-p)}{np(1-p)}\\\\
\\mathrm{Median}(X)=\\lfloor{np}\\rfloor \\vee \\lceil{np}\\rceil\\\\
\\mathrm{Mode}(X)=\\lfloor (n + 1)p \\rfloor \\vee \\lceil (n + 1)p \\rceil - 1\\\\
u:\\text{Uniform[0,1] random varible}\\\\
\\lfloor{x}\\rfloor: \\text{Floor function}\\\\
\\lceil{x}\\rceil: \\text{Ceiling Function}\\\\
I\\left(x,a,b\\right):\\text{Regularized incomplete beta function}\\\\
S(a,b):\\text{Stirling numbers of the second kind}=\\frac1{b!}\\sum_{j=0}^b(-1)^{b-j}\\binom {b}{j} j^a\\\\

GEOMETRIC
X\\sim\\mathrm{Geometric}\\left(p\\right)\\\\
x\\in\\mathbb{N}\\equiv \\left\\{0,1,2,\\dots\\right\\}\\\\
p\\in\\left(0,1\\right)\\subseteq\\mathbb{R}\\\\
F_{X}\\left(x\\right)=1-(1 - p)^{\\lfloor x\\rfloor}\\\\
f_{X}\\left(x\\right)=(1 - p)^{x-1}p\\\\
F^{-1}_{X}\\left(u\\right)=\\left\\lceil{\\frac{\\ln{(1-u)}}{\\ln{(1-p)}}}\\right\\rceil\\\\
E[X^k]=\\mu'_{k}=\\sum_{x=0}^{\\infty}x^{k}f_{X}\\left(x\\right)=\\sum_{x=0}^\\infty (1-p)^x p\\cdot x^k\\\\
\\mathrm{Mean}(X)=\\mu'_{1}=\\frac{1}{p}\\\\
\\mathrm{Variance}(X)=(\\mu'_{2}-\\mu'^{2}_{1})=\\frac{1-p}{p^2}\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}=\\frac{2-p}{\\sqrt{1-p}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}=9+\\frac{p^2}{1-p}\\\\
\\mathrm{Median}(X)=\\left\\lceil \\frac{-1}{\\log_2(1-p)} \\right\\rceil\\\\
\\mathrm{Mode}(X)=1\\\\
u:\\text{Uniform[0,1] random varible}\\\\
\\lfloor{x}\\rfloor: \\text{Floor function}\\\\
\\lceil{x}\\rceil: \\text{Ceiling Function}\\\\

HYPERGEOMETRIC
X\\sim\\mathrm{Hypergeometric}\\left(N,K,n\\right)\\\\
x\\in\\left\\{\\max{(0,n+K-N)} \\dots \\min{(n, K )}\\right\\}\\ \\\\
N\\in\\mathbb{N};K\\in\\left\\{0\\dots N\\right\\};n\\in\\left\\{0\\dots N\\right\\}\\\\
F_{X}\\left(x\\right)=\\sum_{i=0}^{x}\\binom{K}{i}\\binom{N-K}{n-i}\\bigg/{N \\choose n}\\\\
f_{X}\\left(x\\right)=\\binom{K}{x}\\binom{N-K}{n-x}\\bigg/{N \\choose n}\\\\
F^{-1}_{X}\\left(u\\right)=\\arg\\min_{x}\\left| F_{X}\\left(x\\right)-u \\right|\\\\
E[X^k]=\\mu'_{k}=\\sum_{x=\\max{(0,n+K-N)}}^{\\min{(n, K )}}x^{k}f_{X}\\left(x\\right)\\\\
\\mathrm{Mean}(X)=\\mu'_{1}=\\frac{nK}{N}\\\\
\\mathrm{Variance}(X)=(\\mu'_{2}-\\mu'^{2}_{1})=n{K\\over N}{N-K\\over N}{N-n\\over N-1}\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}=\\frac{(N-2K)(N-1)^\\frac{1}{2}(N-2n)}{[nK(N-K)(N-n)]^\\frac{1}{2}(N-2)}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}=3+\\frac{1}{n K(N-K)(N-n)(N-2)(N-3)}\\\\
\\mathrm{Median}(X)=F^{-1}_{X}\\left(0.5\\right)\\\\
\\mathrm{Mode}(X)=\\left \\lfloor \\frac{(n+1)(K+1)}{N+2} \\right \\rfloor\\\\
u:\\text{Uniform[0,1] random varible}\\\\
\\lfloor{x}\\rfloor: \\text{Floor function}\\\\
\\lceil{x}\\rceil: \\text{Ceiling Function}\\\\

LOGARITHMIC
X\\sim\\mathrm{Logarithmic}\\left(p\\right)\\\\
x\\in\\mathbb{N}_{\\geqslant 1}\\equiv \\left\\{1,2,\\dots\\right\\}\\\\
p\\in\\left(0,1\\right)\\subseteq\\mathbb{R}\\\\
F_{X}\\left(x\\right)=\\sum_{i=0}^{x}\\frac{1}{-\\ln(1 - p)} \\frac{p^i}{i}\\\\
f_{X}\\left(x\\right)=\\frac{1}{-\\ln(1 - p)} \\frac{p^x}{x}\\\\
F^{-1}_{X}\\left(u\\right)=\\arg\\min_{x}\\left| F_{X}\\left(x\\right)-u \\right|\\\\
E[X^k]=\\mu'_{k}=\\sum_{x=0}^{\\infty}x^{k}f_{X}\\left(x\\right)=\\frac{(k - 1)!}{-\\ln(1 - p)} \\left(\\frac{p}{1 - p}\\right)^k\\\\
\\mathrm{Mean}(X)=\\mu'_{1}=\\frac{1}{-\\ln(1 - p)} \\frac{p}{1 - p}\\\\
\\mathrm{Variance}(X)=(\\mu'_{2}-\\mu'^{2}_{1})=-\\frac{p^2 + p\\ln(1-p)}{(1-p)^2(\\ln(1-p))^2}\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}\\\\
\\mathrm{Median}(X)=F^{-1}_{X}\\left(0.5\\right)\\\\
\\mathrm{Mode}(X)=1\\\\
u:\\text{Uniform[0,1] random varible}\\\\

NEGATIVE_BINOMIAL
X\\sim\\mathrm{NegativeBinomial}\\left(r,p\\right)\\\\
x\\in\\mathbb{N}\\equiv \\left\\{0,1,2,\\dots\\right\\}\\\\
r\\in\\mathbb{N}_{\\geqslant 1};p\\in\\left(0,1\\right)\\subseteq\\mathbb{R}\\\\
F_{X}\\left(x\\right)=I(p,r,x+1)\\\\
f_{X}\\left(x\\right)=\\binom{r+x-1}{x}p^r(1-p)^x\\\\
F^{-1}_{X}\\left(u\\right)=\\arg\\min_{x}\\left| F_{X}\\left(x\\right)-u \\right|\\\\
E[X^k]=\\mu'_{k}=\\sum_{x=0}^{\\infty}x^{k}f_{X}\\left(x\\right)\\\\
\\mathrm{Mean}(X)=\\mu'_{1}=\\frac{r(1-p)}{p}\\\\
\\mathrm{Variance}(X)=(\\mu'_{2}-\\mu'^{2}_{1})=\\frac{r(1-p)}{p^2}\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}=\\frac{2-p}{\\sqrt{r\\,(1-p)}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}=3+\\frac{6}{r} + \\frac{p^2}{r\\,(1-p)}\\\\
\\mathrm{Median}(X)=F^{-1}_{X}\\left(0.5\\right)\\\\
\\mathrm{Mode}(X)=\\lfloor(r-1)\\,(1-p)/p\\rfloor\\\\
u:\\text{Uniform[0,1] random varible}\\\\
I\\left(x,a,b\\right):\\text{Regularized incomplete beta function}\\\\
\\lfloor{x}\\rfloor: \\text{Floor function}\\\\

POISSON
X\\sim\\mathrm{Poisson}\\left(\\lambda\\right)\\\\
x\\in\\mathbb{N}\\equiv \\left\\{0,1,2,\\dots\\right\\}\\\\
\\lambda\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=e^{-\\lambda} \\sum_{i=0}^{x} \\frac{\\lambda^i}{i!}=1-\\frac{\\gamma(x+1, \\lambda)}{x!}=1-P(x-1,\\lambda)\\\\
f_{X}\\left(x\\right)=\\frac{\\lambda^x e^{-\\lambda}}{x!}\\\\
F^{-1}_{X}\\left(u\\right)=\\arg\\min_{x}\\left| F_{X}\\left(x\\right)-u \\right|\\\\
E[X^k]=\\mu'_{k}=\\sum_{x=0}^{\\infty}x^{k}f_{X}\\left(x\\right)\\\\
\\mathrm{Mean}(X)=\\mu'_{1}=\\lambda\\\\
\\mathrm{Variance}(X)=(\\mu'_{2}-\\mu'^{2}_{1})=\\lambda\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}=\\lambda^{-1/2}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}=3+\\lambda^{-1}\\\\
\\mathrm{Median}(X)=\\lfloor\\lambda+1/3-0.02/\\lambda\\rfloor\\\\
\\mathrm{Mode}(X)=\\lfloor\\lambda\\rfloor\\\\
u:\\text{Uniform[0,1] random varible}\\\\
\\lfloor{x}\\rfloor: \\text{Floor function}\\\\
P\\left(a,x \\right)=\\frac{\\gamma(a,x)}{\\Gamma(a)}:\\text{Regularized lower incomplete gamma function}\\\\
\\gamma\\left(a,x \\right):\\text{Lower incomplete Gamma function}\\\\

UNIFORM
X\\sim\\mathrm{Uniform}\\left(a,b\\right)\\\\
x\\in \\{a,a+1,\\dots,b-1,b\\}\\\\
a\\in\\mathbb{N};b\\in\\mathbb{N};a < b\\\\
F_{X}\\left(x\\right)=\\frac{x -a+1}{b-a+1}\\\\
f_{X}\\left(x\\right)=\\frac{1}{b-a+1}\\\\
F^{-1}_{X}\\left(u\\right)=\\left\\lceil u(b-a+1)+a-1 \\right\\rceil\\\\
E[X^k]=\\mu'_{k}=\\sum_{x=a}^{b}x^{k}f_{X}\\left(x\\right)=\\frac{1}{b-a+1}\\sum_{x=a}^{b}x^{k}\\\\
\\mathrm{Mean}(X)=\\mu'_{1}=\\frac{a+b}{2}\\\\
\\mathrm{Variance}(X)=(\\mu'_{2}-\\mu'^{2}_{1})=\\frac{(b-a+1)^2-1}{12}\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}=0\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}=3-\\frac{6((b-a+1)^2+1)}{5((b-a+1)^2-1)}\\\\
\\mathrm{Median}(X)=\\frac{a+b}{2}\\\\
\\mathrm{Mode}(X)\\in [a, b]\\\\
u:\\text{Uniform[0,1] random varible}\\\\
\\lceil{x}\\rceil: \\text{Ceiling Function}\\\\
`;