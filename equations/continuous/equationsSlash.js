const eqs = `
ALPHA
X\\sim\\mathrm{Alpha}\\left(\\alpha,L,S\\right)\\\\
x\\in\\mathbb{R}^{+}\\\\
\\alpha\\in\\mathbb{R}^{+},L\\in\\mathbb{R},S\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=\\frac{\\Phi\\left(\\alpha-\\frac{1}{z(x)}\\right)}{\\Phi\\left(\\alpha\\right)}\\\\
f_{X}\\left(x\\right)=\\frac{1}{S\\cdot z(x)^{2}\\cdot \\Phi\\left(\\alpha\\right)\\cdot\\sqrt{2\\pi}}\\exp\\left(-\\frac{1}{2}\\left(\\alpha-\\frac{1}{z(x)}\\right)^{2}\\right)\\\\
F_{X}^{-1}\\left(u\\right)=L+S\\times \\frac{1}{\\alpha-\\Phi^{-1}\\left(u\\Phi\\left(\\alpha\\right)\\right)}\\\\
\\tilde{\\mu}'_{k}=E[\\tilde{X}^k]=\\int_{0}^{\\infty}x^{k}f_{\\tilde{X}}\\left(x\\right)dx\\\\
\\mathrm{Mean}(X)=L+S\\cdot \\tilde{\\mu}'_{1}\\\\
\\mathrm{Variance}(X)=S^{2}\\cdot(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})\\\\
\\mathrm{Skewness}(X)=\\frac{\\tilde{\\mu}'_{3}-3\\tilde{\\mu}'_{2}\\tilde{\\mu}'_{1}+2\\tilde{\\mu}'^{3}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{1.5}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\tilde{\\mu}'_{4}-4\\tilde{\\mu}'_{1}\\tilde{\\mu}'_{3}+6\\tilde{\\mu}'^{2}_{1}\\tilde{\\mu}'_{2}-3\\tilde{\\mu}'^{4}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{2}}\\\\
\\mathrm{Median}(X)=L+\\frac{S}{\\alpha-\\Phi^{-1}\\left(\\frac{1}{2}\\Phi\\left(\\alpha\\right)\\right)}\\\\
\\mathrm{Mode}(X)=L+S\\frac{(\\sqrt{\\alpha^{2}+8}-\\alpha)}{4}\\\\
\\tilde{X}\\sim \\mathrm{Alpha}\\left(\\alpha,0,1\\right)\\\\
L:\\text{Location parameter}\\\\
S:\\text{Scale parameter}\\\\
z(x)=\\left(x-L\\right)/S\\\\
u:\\text{Uniform[0,1] random varible}\\\\
\\Phi\\left(x\\right):\\text{Cumulative function from standard normal distribution}\\\\
\\Phi^{-1}\\left(x\\right):\\text{Inverse of cumulative function from standard normal distribution}\\\\

ARCSINE
X\\sim\\mathrm{Arcsine}\\left(a,b\\right)\\\\
x\\in\\left(a,b\\right)\\\\
a\\in\\mathbb{R},b\\in\\mathbb{R},a\\lt b\\\\
F_{X}(x)=\\frac{2}{\\pi}\\arcsin\\left(\\sqrt\\frac{x-a}{b-a}\\right)\\\\
f_{X}(x)=\\frac{1}{\\pi\\sqrt{(x-a)(b-x)}}\\\\
F_{X}^{-1}\\left(u\\right)=a+\\left(b-a\\right)\\times \\sin^{2}\\left(\\frac{\\pi}{2}u\\right)\\\\
\\tilde{\\mu}'_{k}=E[\\tilde{X}^k]=\\int_{0}^{1}x^{k}f_{\\tilde{X}}\\left(x\\right)dx=\\frac{1}{\\pi}Beta\\left(\\frac{1}{2},k+\\frac{1}{2}\\right)=\\frac{\\left(2k-1\\right)!!}{2^{k}k!}\\\\
\\mathrm{Mean}(X)=a+\\tilde{\\mu}'_{1}\\left(b-a\\right)= a+\\frac{1}{2}\\left(b-a\\right)\\\\
\\mathrm{Variance}(X)=\\left(b-a\\right)^{2}\\times (\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})=\\frac{\\left(b-a\\right)^{2}}{8}\\\\
\\mathrm{Skewness}(X)=\\frac{\\tilde{\\mu}'_{3}-3\\tilde{\\mu}'_{2}\\tilde{\\mu}'_{1}+2\\tilde{\\mu}'^{3}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{1.5}}=0\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\tilde{\\mu}'_{4}-4\\tilde{\\mu}'_{1}\\tilde{\\mu}'_{3}+6\\tilde{\\mu}'^{2}_{1}\\tilde{\\mu}'_{2}-3\\tilde{\\mu}'^{4}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{2}}=3-\\frac{3}{2}\\\\
\\mathrm{Median}(X)=a+\\left(b-a\\right)\\times \\sin^{2}\\left(\\frac{\\pi}{4}\\right)\\\\
\\mathrm{Mode}(X)=\\text{undefined}\\\\
\\tilde{X}\\sim \\mathrm{Arcsine}\\left(0,1\\right)\\\\
Beta\\left(x,y\\right):\\text{Beta function}\\\\
u:\\text{Uniform[0,1] random varible}\\\\

ARGUS
X\\sim\\mathrm{Argus}\\left(\\chi,L,S\\right)\\\\
x\\in\\left(L,L+S\\right)\\\\
\\chi\\in\\mathbb{R}^{+},L\\in\\mathbb{R},S\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=1-\\frac{\\Psi\\left(\\chi\\sqrt{1-z(x)^2}\\right)}{\\Psi(\\chi)}\\\\
f_{X}\\left(x\\right)=\\frac{1}{S}\\cdot\\frac{\\chi^3}{\\sqrt{2\\pi}\\,\\Psi(\\chi)}\\cdot z(x)\\sqrt{1-z(x)^2}\\exp\\left(-\\frac12 \\chi^2\\Big(1-z(x)^2\\Big)\\right)\\\\
F^{-1}_{X}\\left(u\\right)=L+S\\sqrt{1-\\frac{2P^{-1}(\\frac{3}{2},(1-u)P(\\frac{3}{2},\\frac{\\chi^{2}}{2}))}{\\chi^{2}}}\\\\
\\mu'_{k}=E[X^k]=\\int_{L}^{L+S}x^{k}f_{X}\\left(x\\right)dx\\\\
\\mathrm{Mean}(X)=\\mu'_{1}=L+S\\sqrt{\\pi/8}\\,\\frac{\\chi e^{-\\frac{\\chi^2}{4}} I_1(\\tfrac{\\chi^2}{4})}{\\Psi(\\chi)}\\\\
\\mathrm{Variance}(X)=\\mu'_{2}-\\mu'^{2}_{1}=S^2\\cdot\\left(1-\\frac{3}{\\chi^2}+\\frac{\\chi\\phi(\\chi)}{\\Psi(\\chi)}\\right)-(\\mu-L)^2\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}\\\\
\\mathrm{Median}(X)=L+S\\sqrt{1-\\frac{2P^{-1}(\\frac{3}{2},\\frac{1}{2}P(\\frac{3}{2},\\frac{\\chi^{2}}{2}))}{\\chi^{2}}}\\\\
\\mathrm{Mode}(X)=L+\\frac{S}{\\sqrt2\\chi}\\sqrt{(\\chi^2-2)+\\sqrt{\\chi^4+4}}\\\\
L:\\text{Location parameter}\\\\
S:\\text{Scale parameter}\\\\
z(x)=\\left(x-L\\right)/S\\\\
u:\\text{Uniform[0,1] random varible}\\\\
\\Psi(\\chi)=\\Phi(\\chi)- \\chi \\phi( \\chi )-\\tfrac{1}{2}\\\\
\\Phi(x):\\text{Cumulative function standard normal distribution}\\\\
\\phi(x):\\text{Density function from standard normal distribution}\\\\
I_{\\alpha}\\left(x\\right):\\text{Modified Bessel function of the first kind of order }\\alpha\\in\\mathbb{N}\\\\
P(a,x)=\\frac{\\gamma(a,x)}{\\Gamma(a)}:\\text{Regularized lower incomplete gamma function}\\\\
P^{-1}(a,y):\\text{Inverse of regularized lower incomplete gamma function}\\\\
\\gamma(a,x):\\text{Lower incomplete gamma function}\\\\
\\Gamma(x):\\text{Gamma function}\\\\

BETA
X\\sim\\mathrm{Beta}\\left(\\alpha,\\beta,A,B\\right)\\\\
x\\in\\left(A,B\\right)\\\\
\\alpha\\in\\mathbb{R}^{+},\\beta\\in\\mathbb{R}^{+},A\\in\\mathbb{R},B\\in\\mathbb{R},A\\lt B\\\\
F_{X}\\left(x\\right)=I\\left(z(x),\\alpha,\\beta\\right)\\\\
f_{X}\\left(x\\right)=\\frac{z(x)^{\\alpha-1}\\left(1-z(x)\\right)^{\\beta-1}}{Beta(\\alpha,\\beta)(B-A)}\\\\
F^{-1}_{X}\\left(u\\right)=A+(B-A)\\times I^{-1}\\left(u,\\alpha,\\beta\\right)\\\\
\\tilde{\\mu}'_{k}=E[\\tilde{X}^k]=\\int_{0}^{1}x^{k}f_{\\tilde{X}}\\left(x\\right)dx\\\\
\\mathrm{Mean}(X)=A+\\left(B-A\\right)\\cdot\\tilde{\\mu}'_{1}=A+\\frac{\\alpha\\left(B-A\\right)}{\\alpha+\\beta}\\\\
\\mathrm{Variance}(X)=\\left(B-A\\right)^{2}\\cdot(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})=\\frac{\\alpha\\beta\\left(B-A\\right)^{2}}{(\\alpha+\\beta)^2(\\alpha+\\beta+1)}\\\\
\\mathrm{Skewness}(X)=\\frac{\\tilde{\\mu}'_{3}-3\\tilde{\\mu}'_{2}\\tilde{\\mu}'_{1}+2\\tilde{\\mu}'^{3}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{1.5}}=\\frac{2\\,(\\beta-\\alpha)\\sqrt{\\alpha+\\beta+1}}{(\\alpha+\\beta+2)\\sqrt{\\alpha\\beta}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\tilde{\\mu}'_{4}-4\\tilde{\\mu}'_{1}\\tilde{\\mu}'_{3}+6\\tilde{\\mu}'^{2}_{1}\\tilde{\\mu}'_{2}-3\\tilde{\\mu}'^{4}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{2}}=3+\\frac{6[(\\alpha-\\beta)^2 (\\alpha +\\beta+1)-\\alpha \\beta (\\alpha+\\beta+2)]}{\\alpha \\beta (\\alpha+\\beta+2) (\\alpha+\\beta+3)}\\\\
\\mathrm{Median}(X)=A+(B-A)\\times I^{-1}\\left(\\frac{1}{2},\\alpha,\\beta\\right) \\quad \\text{if }\\alpha,\\beta>1\\\\
\\mathrm{Mode}(X)=A+(B-A)\\frac{\\alpha-1}{\\alpha+\\beta-2} \\quad \\text{if }\\alpha,\\beta>1\\\\
\\tilde{X}\\sim\\mathrm{Beta}\\left(\\alpha,\\beta,0,1\\right)\\\\
z\\left(x\\right)=\\left(x-A\\right)/\\left(B-A\\right)\\\\
u:\\text{Uniform[0,1] random varible}\\\\
I\\left(x,a,b\\right):\\text{Regularized incomplete beta function}\\\\
I^{-1}\\left(x,a,b\\right):\\text{Inverse of regularized incomplete beta function}\\\\
Beta\\left(x,y\\right):\\text{Beta function}\\\\  

BETA_PRIME
X\\sim\\mathrm{BetaPrime}\\left(\\alpha,\\beta\\right)\\\\
x\\in [0,\\infty)\\\\
\\alpha\\in\\mathbb{R}^{+},\\beta\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=I\\left(\\frac{x}{1+x},\\alpha,\\beta\\right)\\\\
f_{X}\\left(x\\right)=\\frac{x^{\\alpha-1} (1+x)^{-\\alpha -\\beta}}{Beta(\\alpha,\\beta)}\\\\
F^{-1}_{X}\\left(u\\right)=\\frac{I^{-1}\\left(u,\\alpha,\\beta\\right)}{1-I^{-1}\\left(u,\\alpha,\\beta\\right)}\\\\
\\mu'_{k}=E[X^k]=\\int_{0}^{\\infty}x^{k}f_{X}\\left(x\\right)dx=\\frac{\\Gamma\\left(k+\\alpha\\right)\\Gamma\\left(\\beta-k\\right)}{\\Gamma\\left(\\alpha\\right)\\Gamma\\left(\\beta\\right)} \\quad \\text{if }\\beta>k\\\\
\\mathrm{Mean}(X)=\\mu'_{1}=\\frac{\\alpha}{\\beta-1} \\quad \\text{if }\\beta>1\\\\
\\mathrm{Variance}(X)=\\mu'_{2}-\\mu'^{2}_{1}=\\frac{\\alpha(\\alpha+\\beta-1)}{(\\beta-2)(\\beta-1)^2} \\quad \\text{if }\\beta>2\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}=\\frac{2(2\\alpha+\\beta-1)}{\\beta-3}\\sqrt{\\frac{\\beta-2}{\\alpha(\\alpha+\\beta-1)}} \\quad \\text{if }\\beta>3\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}} \\quad \\text{if }\\beta>4\\\\
\\mathrm{Median}(X)=\\frac{I^{-1}\\left(\\frac{1}{2},\\alpha,\\beta\\right)}{1-I^{-1}\\left(\\frac{1}{2},\\alpha,\\beta\\right)}\\\\
\\mathrm{Mode}(X)=\\frac{\\alpha-1}{\\beta+1}\\\\
u:\\text{Uniform[0,1] random varible}\\\\
I\\left(x,a,b\\right):\\text{Regularized incomplete beta function}\\\\
I^{-1}\\left(x,a,b\\right):\\text{Inverse of regularized incomplete beta function}\\\\
\\Gamma\\left(x\\right):\\text{Gamma function}\\\\
Beta\\left(x,y\\right):\\text{Beta function}\\\\

BETA_PRIME_4P
X\\sim\\mathrm{BetaPrime}_{\\mathrm{4P}}\\left(\\alpha,\\beta,L,S\\right)\\\\
x\\in [L,\\infty)\\\\
\\alpha\\in\\mathbb{R}^{+},\\beta\\in\\mathbb{R}^{+},L\\in\\mathbb{R},S\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=I\\left(\\frac{z(x)}{1+z(x)},\\alpha,\\beta\\right) \\\\
f_{X}\\left(x\\right)=\\frac{z(x)^{\\alpha-1} (1+z(x))^{-\\alpha -\\beta}}{S\\times Beta(\\alpha,\\beta)}\\\\
F^{-1}_{X}\\left(u\\right)=L+S\\frac{I^{-1}\\left(u,\\alpha,\\beta\\right)}{1-I^{-1}\\left(u,\\alpha,\\beta\\right)}\\\\
\\tilde{\\mu}'_{k}=E[\\tilde{X}^k]=\\int_{0}^{\\infty}x^{k}f_{\\tilde{X}}\\left(x\\right)dx=\\frac{\\Gamma\\left(k+\\alpha\\right)\\Gamma\\left(\\beta-k\\right)}{\\Gamma\\left(\\alpha\\right)\\Gamma\\left(\\beta\\right)} \\quad \\text{if }\\beta>k\\\\
\\mathrm{Mean}(X)=L+S\\tilde{\\mu}'_{1}=L+S\\frac{\\alpha}{\\beta-1} \\quad \\text{if }\\beta>1\\\\
\\mathrm{Variance}(X)=S^{2}(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})=S^{2}\\frac{\\alpha(\\alpha+\\beta-1)}{(\\beta-2)(\\beta-1)^2} \\quad \\text{if }\\beta>2\\\\
\\mathrm{Skewness}(X)=\\frac{\\tilde{\\mu}'_{3}-3\\tilde{\\mu}'_{2}\\tilde{\\mu}'_{1}+2\\tilde{\\mu}'^{3}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{1.5}}=\\frac{2(2\\alpha+\\beta-1)}{\\beta-3}\\sqrt{\\frac{\\beta-2}{\\alpha(\\alpha+\\beta-1)}} \\quad \\text{if }\\beta>3\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\tilde{\\mu}'_{4}-4\\tilde{\\mu}'_{1}\\tilde{\\mu}'_{3}+6\\tilde{\\mu}'^{2}_{1}\\tilde{\\mu}'_{2}-3\\tilde{\\mu}'^{4}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{2}} \\quad \\text{if }\\beta>4\\\\
\\mathrm{Median}(X)=L+S\\frac{I^{-1}\\left(\\frac{1}{2},\\alpha,\\beta\\right)}{1-I^{-1}\\left(\\frac{1}{2},\\alpha,\\beta\\right)}\\\\
\\mathrm{Mode}(X)=L+S\\frac{\\alpha-1}{\\beta+1}\\\\
\\tilde{X}\\sim \\mathrm{BetaPrime}\\left( \\alpha,\\beta \\right)\\\\
L:\\text{Location parameter}\\\\
S:\\text{Scale parameter}\\\\
z\\left(x\\right)=\\left(x-L\\right)/S\\\\
u:\\text{Uniform[0,1] random varible}\\\\
I\\left(x,a,b\\right):\\text{Regularized incomplete beta function}\\\\
I^{-1}\\left(x,a,b\\right):\\text{Inverse of regularized incomplete beta function}\\\\
\\Gamma\\left(x\\right):\\text{Gamma function}\\\\
Beta\\left(x,y\\right):\\text{Beta function}\\\\

BRADFORD
X\\sim\\mathrm{Bradford}\\left(c,min,max\\right)\\\\
x\\in\\left(min,max\\right)\\\\
c\\in\\mathbb{R}^{+},min\\in\\mathbb{R},max\\in\\mathbb{R},min\\lt max\\\\
F_{X}\\left(x\\right)=\\frac{\\ln\\left(1+c\\cdot z(x)\\right)}{k}\\\\
f_{X}\\left(x\\right)=\\frac{c}{k\\left(1+c\\cdot z(x)\\right)\\left(max-min\\right)}\\\\
F^{-1}_{X}\\left(u\\right)=min+(max-min)\\times \\frac{\\left(1+c\\right)^{u}-1}{c}\\\\
\\tilde{\\mu}'_{k}=E[\\tilde{X}^k]=\\int_{0}^{1}x^{k}f_{\\tilde{X}}\\left(x\\right)dx\\\\
\\mathrm{Mean}(X)=min+\\left(max-min\\right)\\cdot\\tilde{\\mu}'_{1}=min+\\left(max-min\\right)\\cdot\\frac{c-k}{ck}\\\\
\\mathrm{Variance}(X)=\\left(max-min\\right)^{2}\\cdot(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})=\\left(max-min\\right)^{2}\\cdot\\frac{\\left(c+2\\right)k-2c}{2ck^{2}}\\\\
\\mathrm{Skewness}(X)=\\frac{\\tilde{\\mu}'_{3}-3\\tilde{\\mu}'_{2}\\tilde{\\mu}'_{1}+2\\tilde{\\mu}'^{3}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{1.5}}=\\frac{\\sqrt{2}\\left(12c^{2}-9kc\\left(c+2\\right)+2k^{2}\\left(c\\left(c+3\\right)+3\\right)\\right)}{\\sqrt{c\\left(c\\left(k-2\\right)+2k\\right)}\\left(3c\\left(k-2\\right)+6k\\right)}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\tilde{\\mu}'_{4}-4\\tilde{\\mu}'_{1}\\tilde{\\mu}'_{3}+6\\tilde{\\mu}'^{2}_{1}\\tilde{\\mu}'_{2}-3\\tilde{\\mu}'^{4}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{2}}=3+\\frac{c^{3}\\left(k-3\\right)\\left(k\\left(3k-16\\right)+24\\right)+12kc^{2}\\left(k-4\\right)\\left(k-3\\right)+6ck^{2}\\left(3k-14\\right)+12k^{3}}{3c\\left(c\\left(k-2\\right)+2k\\right)^{2}}\\\\
\\mathrm{Median}(X)=min+(max-min)\\cdot\\frac{\\left(1+c\\right)^{frac{1}{2}}-1}{c}\\\\
\\mathrm{Mode}(X)=min\\\\
\\tilde{X}\\sim\\mathrm{Bradford}\\left(c,0,1\\right)\\\\
k=\\ln(1+c)\\\\
z\\left(x\\right)=\\left(x-min\\right)/\\left(max-min\\right)\\\\
u:\\text{Uniform[0,1] random varible}\\\\

BURR
X\\sim\\mathrm{Burr}\\left(A,B,C\\right)\\\\
x\\in\\left[0,\\infty\\right)\\\\
A\\in\\mathbb{R}^{+},B\\in\\mathbb{R},C\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=1-\\left[1+\\left(\\frac{x}{A}\\right)^{B}\\right]^{-C}\\\\
f_{X}\\left(x\\right)=\\frac{BC}{A}\\left(\\frac{x}{A}\\right)^{B-1}\\left[1+\\left(\\frac{x}{A}\\right)^{B}\\right]^{-C-1}\\\\
F^{-1}_{X}\\left(u\\right)=A\\left[(1-u)^{-\\frac{1}{c}}-1\\right]^{\\frac{1}{B}}\\\\
\\mu'_{k}=E[X^k]=\\int_{0}^{\\infty}x^{k}f_{X}\\left(x\\right)dx=A^{k}C\\times Beta\\left(\\frac{BC-k}{B},\\frac{B+K}{B}\\right)\\\\
\\mathrm{Mean}(X)=\\mu'_{1}\\\\
\\mathrm{Variance}(X)=\\mu'_{2}-\\mu'^{2}_{1}\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}\\\\
\\mathrm{Median}(X)=A\\left[\\left(\\frac{1}{2}\\right)^{-\\frac{1}{c}}-1\\right]^{\\frac{1}{B}}\\\\
\\mathrm{Mode}(X)=A\\left(\\frac{B-1}{BC+1}\\right)^{\\frac{1}{B}}\\\\
u:\\text{Uniform[0,1] random varible}\\\\
Beta\\left(x,y\\right):\\text{Beta function}\\\\

BURR_4P
X\\sim \\mathrm{Burr_{4P}}\\left( A,B,C,L \\right)\\\\
x\\in \\left[ L,\\infty \\right)\\\\
A\\in \\mathbb{R}^{+}, B\\in \\mathbb{R}, C\\in \\mathbb{R}^{+}, L\\in \\mathbb{R}\\\\
F_{X}\\left( x \\right)=1-\\left[ 1+\\left( \\frac{x-L}{A} \\right)^{B} \\right]^{-C}\\\\
f_{X}\\left( x \\right)=\\frac{BC}{A}\\left( \\frac{x-L}{A} \\right)^{B-1}\\left[ 1+\\left( \\frac{x-L}{A} \\right)^{B} \\right]^{-C-1}\\\\
F^{-1}_{X}\\left( u \\right)=L+A\\left[ (1-u)^{-\\frac{1}{c}}-1 \\right]^{\\frac{1}{B}}\\\\
\\tilde{\\mu}'_{k}=E[\\tilde{X}^k]=\\int_{0}^{\\infty}x^{k}f_{\\tilde{X}}=A^{k}C\\times Beta\\left( \\frac{BC-k}{B},\\frac{B+K}{B} \\right)\\\\
\\mathrm{Mean}(X)=L+\\tilde{\\mu}'_{1}\\\\
\\mathrm{Variance}(X)=\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1}\\\\
\\mathrm{Skewness}(X) = \\frac{\\tilde{\\mu}'_{3} - 3\\tilde{\\mu}'_{2}\\tilde{\\mu}'_{1} + 2\\tilde{\\mu}'^{3}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{1.5}}\\\\
\\mathrm{Kurtosis}(X) = \\frac{\\tilde{\\mu}'_{4} - 4\\tilde{\\mu}'_{1}\\tilde{\\mu}'_{3} + 6\\tilde{\\mu}'^{2}_{1}\\tilde{\\mu}'_{2} - 3\\tilde{\\mu}'^{4}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{2}}\\\\
\\mathrm{Median}(X)=L+A\\left[\\left(\\frac{1}{2}\\right)^{-\\frac{1}{c}}-1 \\right]^{\\frac{1}{B}}\\\\
\\mathrm{Mode}(X)=L+A\\left( \\frac{B-1}{BC+1} \\right)^{\\frac{1}{B}}\\\\
\\tilde{X}\\sim \\mathrm{Burr}\\left( A,B,C \\right)\\\\
L:\\text{Location parameter}\\\\
u:\\text{Uniform[0,1] random varible}\\\\
Beta\\left( x,y \\right):\\text{Beta function}\\\\

CAUCHY
X\\sim\\mathrm{Cauchy}\\left(x0,\\gamma\\right)\\\\
x\\in (-\\infty,+\\infty)\\\\
x_0\\in\\mathbb{R},\\gamma\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=\\frac{1}{\\pi} \\arctan\\left(\\frac{x-x_0}{\\gamma}\\right)+\\frac{1}{2}\\\\
f_{X}\\left(x\\right)=\\frac{1}{\\pi\\gamma\\,\\left[1+\\left(\\frac{x-x_0}{\\gamma}\\right)^2\\right]}\\\\
F^{-1}_{X}\\left(u\\right)=x_0+\\gamma\\,\\tan\\left[\\pi\\left(u-\\tfrac{1}{2}\\right)\\right]\\\\
\\mu'_{k}=E[X^k]=\\int_{-\\infty }^{\\infty }x^{k}f_{X}\\left(x\\right)dx\\\\
\\mathrm{Mean}(X)=\\text{undefined}\\\\
\\mathrm{Variance}(X)=\\text{undefined}\\\\
\\mathrm{Skewness}(X)=\\text{undefined}\\\\
\\mathrm{Kurtosis}(X)=\\text{undefined}\\\\
\\mathrm{Median}(X)=x_0\\\\
\\mathrm{Mode}(X)=x_0\\\\
x_0:\\text{Location parameter}\\\\
\\gamma:\\text{Scale parameter}\\\\
u:\\text{Uniform[0,1] random varible}\\\\

CHI_SQUARE
X\\sim\\mathrm{\\chi^{2}}\\left(df\\right)\\\\
x\\in\\left(0,\\infty\\right)\\\\
df\\in\\mathbb{N}^{+}\\\\
F_{X}\\left(x\\right)=\\frac{\\gamma(\\frac{df}{2},\\,\\frac{x}{2})}{\\Gamma(\\frac{df}{2})}=P\\left(\\frac{df}{2},\\,\\frac{x}{2}\\right)\\\\
f_{X}\\left(x\\right)=\\frac{1}{2^{df/2}\\Gamma(df/2)}\\,x^{df/2-1} e^{-x/2}\\\\
F^{-1}_{X}\\left(u\\right)=2P^{-1}\\left(\\frac{df}{2},u\\right)\\\\
\\mu'_{k}=E[X^k]=\\int_{0}^{\\infty }x^{k}f_{X}\\left(x\\right)dx= df(df+2)\\cdots(df+2k-2)=2^k\\frac{\\Gamma\\left(k+\\frac{df}{2}\\right)}{\\Gamma\\left(\\frac{df}{2}\\right)}\\\\
\\mathrm{Mean}(X)=\\mu'_{1}=df\\\\
\\mathrm{Variance}(X)=\\mu'_{2}-\\mu'^{2}_{1}=2df\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}=\\sqrt{8/df}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}=3+\\frac{12}{df}\\\\
\\mathrm{Median}(X)=2P^{-1}\\left(\\frac{df}{2},\\frac{1}{2}\\right)\\\\
\\mathrm{Mode}(X)=\\max(df-2,0)\\\\
u:\\text{Uniform[0,1] random varible}\\\\
P\\left(a,x\\right)=\\frac{\\gamma(a,x)}{\\Gamma(a)}:\\text{Regularized lower incomplete gamma function}\\\\
P^{-1}\\left(a,u\\right):\\text{Inverse of regularized lower incomplete gamma function}\\\\
\\gamma\\left(a,x\\right):\\text{Lower incomplete gamma function}\\\\
\\Gamma\\left(x\\right):\\text{Gamma function}\\\\

CHI_SQUARE_3P
X\\sim\\mathrm{\\chi^{2}_{3P}}\\left(df,L,S\\right)\\\\
x\\in\\left(L,\\infty\\right)\\\\
df\\in\\mathbb{N}^{+},L\\in\\mathbb{R},S\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=\\frac{\\gamma(\\frac{df}{2},\\,\\frac{z(x)}{2})}{\\Gamma(\\frac{df}{2})}=P\\left(\\frac{df}{2},\\,\\frac{z(x)}{2}\\right)\\\\
f_{X}\\left(x\\right)=\\frac{1}{S}\\frac{1}{2^{df/2}\\Gamma(df/2)}\\,x^{df/2-1} e^{-z(x)/2}\\\\
F^{-1}_{X}\\left(u\\right)=2P^{-1}\\left(\\frac{df}{2},u\\right)\\\\
\\tilde{\\mu}'_{k}=E[\\tilde{X}^k]=\\int_{0}^{\\infty}x^{k}f_{\\tilde{X}}\\left(x\\right)dx=df(df+2)\\cdots(df+2k-2)=2^k\\frac{\\Gamma\\left(k+\\frac{df}{2}\\right)}{\\Gamma\\left(\\frac{df}{2}\\right)}\\\\
\\mathrm{Mean}(X)=L+S\\cdot \\tilde{\\mu}'_{1}=L+S\\cdot df\\\\
\\mathrm{Variance}(X)=S^{2}\\cdot(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})=2\\cdot df\\cdot S^{2}\\\\
\\mathrm{Skewness}(X)=\\frac{\\tilde{\\mu}'_{3}-3\\tilde{\\mu}'_{2}\\tilde{\\mu}'_{1}+2\\tilde{\\mu}'^{3}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{1.5}}=\\sqrt{8/df}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\tilde{\\mu}'_{4}-4\\tilde{\\mu}'_{1}\\tilde{\\mu}'_{3}+6\\tilde{\\mu}'^{2}_{1}\\tilde{\\mu}'_{2}-3\\tilde{\\mu}'^{4}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{2}}=3+\\frac{12}{df}\\\\
\\mathrm{Median}(X)=L+S\\times 2P^{-1}\\left(\\frac{df}{2},\\frac{1}{2}\\right)\\\\
\\mathrm{Mode}(X)=L+S\\times \\max(df-2,0)\\\\
\\tilde{X}\\sim\\mathrm{\\chi^{2}}\\left(df\\right)\\\\
L:\\text{Location parameter}\\\\
S:\\text{Scale parameter}\\\\
z\\left(x\\right)=\\left(x-L\\right)/S\\\\
u:\\text{Uniform[0,1] random varible}\\\\
P\\left(a,x\\right)=\\frac{\\gamma(a,x)}{\\Gamma(a)}:\\text{Regularized lower incomplete gamma function}\\\\
P^{-1}\\left(a,u\\right):\\text{Inverse of regularized lower incomplete gamma function}\\\\
\\gamma\\left(a,x\\right):\\text{Lower incomplete gamma function}\\\\
\\Gamma\\left(x\\right):\\text{Gamma function}\\\\

DAGUM
X\\sim\\mathrm{Dagum}\\left(a,b,p\\right)\\\\
x\\in\\left(0,\\infty\\right)\\\\
a\\in\\mathbb{R}^{+},b\\in\\mathbb{R}^{+},p\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)={\\left(1+{\\left(\\frac{x}{b}\\right)}^{-a}\\right)}^{-p}\\\\
f_{X}\\left(x\\right)=\\frac{a p}{x}\\left(\\frac{(\\tfrac{x}{b})^{a p}}{\\left((\\tfrac{x}{b})^a+1\\right)^{p+1}}\\right)\\\\
F^{-1}_{X}\\left(u\\right)=b(u^{-1/p}-1)^{-1/a}\\\\
\\mu'_{k}=E[X^k]=\\int_{0}^{\\infty}x^{k}f_{X}\\left(x\\right)dx=pb^{k}\\cdot Beta\\left(\\frac{ap+k}{a},\\frac{a-k}{a}\\right)\\\\
\\mathrm{Mean}(X)=\\mu'_{1}\\\\
\\mathrm{Variance}(X)=\\mu'_{2}-\\mu'^{2}_{1}\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}\\\\
\\mathrm{Median}(X)=b{\\left(-1+2^{\\tfrac{1}{p}}\\right)}^{-\\tfrac{1}{a}}\\\\
\\mathrm{Mode}(X)=b{\\left(\\frac{ap-1}{a+1}\\right)}^{\\tfrac{1}{a}}\\\\
b:\\text{Scale parameter}\\\\
u:\\text{Uniform[0,1] random varible}\\\\
Beta\\left(x,y\\right):\\text{Beta function}\\\\

DAGUM_4P
X\\sim\\mathrm{Dagum_{4P}}\\left(a,b,p,L\\right)\\\\
x\\in\\left(L,\\infty\\right)\\\\
a\\in\\mathbb{R}^{+},b\\in\\mathbb{R}^{+},p\\in\\mathbb{R}^{+},L\\in\\mathbb{R}\\\\
F_{X}\\left(x\\right)={\\left(1+{\\left(\\frac{x-L}{b}\\right)}^{-a}\\right)}^{-p}\\\\
f_{X}\\left(x\\right)=\\frac{a p}{x-L}\\left(\\frac{(\\tfrac{x-L}{b})^{a p}}{\\left((\\tfrac{x-L}{b})^a+1\\right)^{p+1}}\\right)\\\\
F^{-1}_{X}\\left(u\\right)=L+b(u^{-1/p}-1)^{-1/a}\\\\
\\tilde{\\mu}'_{k}=E[\\tilde{X}^k]=\\int_{0}^{\\infty}x^{k}f_{\\tilde{X}}\\left(x\\right)dx=pb^{k}\\cdot Beta\\left(\\frac{ap+k}{a},\\frac{a-k}{a}\\right)\\\\
\\mathrm{Mean}(X)=L+\\tilde{\\mu}'_{1}\\\\
\\mathrm{Variance}(X)=\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1}\\\\
\\mathrm{Skewness}(X)=\\frac{\\tilde{\\mu}'_{3}-3\\tilde{\\mu}'_{2}\\tilde{\\mu}'_{1}+2\\tilde{\\mu}'^{3}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{1.5}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\tilde{\\mu}'_{4}-4\\tilde{\\mu}'_{1}\\tilde{\\mu}'_{3}+6\\tilde{\\mu}'^{2}_{1}\\tilde{\\mu}'_{2}-3\\tilde{\\mu}'^{4}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{2}}\\\\
\\mathrm{Median}(X)=L+b{\\left(-1+2^{\\tfrac{1}{p}}\\right)}^{-\\tfrac{1}{a}}\\\\
\\mathrm{Mode}(X)=L+b{\\left(\\frac{ap-1}{a+1}\\right)}^{\\tfrac{1}{a}}\\\\
\\bar{X}\\sim\\mathrm{Dagum}\\left(a,b,p\\right)\\\\
L:\\text{Location parameter}\\\\
b:\\text{Scale parameter}\\\\
u:\\text{Uniform[0,1] random varible}\\\\
Beta\\left(x,y\\right):\\text{Beta function}\\\\

ERLANG
X\\sim\\mathrm{Erlang}\\left(k,\\beta\\right)\\\\
x\\in\\left[0,\\infty\\right)\\\\
k\\in\\mathbb{N}^{+},\\beta\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=P(k,\\frac{x}{\\beta})=\\frac{\\gamma(k,\\frac{x}{\\beta})}{(k-1)!}\\\\
f_{X}\\left(x\\right)=\\frac{x^{k-1} e^{-\\frac{x}{\\beta}}}{\\beta^k(k-1)!}\\\\
F^{-1}_{X}\\left(u\\right)=\\beta P^{-1}\\left(k,u\\right)\\\\
\\mu'_{n}=E[X^n]=\\int_{0}^{\\infty}x^{n}f_{X}\\left(x\\right)dx=\\beta^{k}\\frac{\\Gamma\\left(n+k\\right)}{\\Gamma(k)}\\\\
\\mathrm{Mean}(X)=\\mu'_{1}\\\\
\\mathrm{Variance}(X)=\\mu'_{2}-\\mu'^{2}_{1}\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}\\\\
\\mathrm{Median}(X)=P(k,\\frac{1}{2\\beta})\\\\
\\mathrm{Mode}(X)=\\beta\\left(k-1\\right)\\\\
\\beta:\\text{Scale parameter}\\\\
u:\\text{Uniform[0,1] random varible}\\\\
P\\left(a,x\\right)=\\frac{\\gamma(a,x)}{\\Gamma(a)}:\\text{Regularized lower incomplete gamma function}\\\\
P^{-1}\\left(a,u\\right):\\text{Inverse of regularized lower incomplete gamma function}\\\\
\\gamma\\left(a,x\\right):\\text{Lower incomplete gamma function}\\\\
\\Gamma\\left(x\\right):\\text{Gamma function}\\\\

ERLANG_3P
X\\sim\\mathrm{Erlang_{3P}}\\left(k,\\beta,L\\right)\\\\
x\\in\\left[L,\\infty\\right)\\\\
k\\in\\mathbb{N}^{+},\\beta\\in\\mathbb{R}^{+},L\\in\\mathbb{R}\\\\
F_{X}\\left(x\\right)=P(k,\\frac{x-L}{\\beta})=\\frac{\\gamma(k,\\frac{x-L}{\\beta})}{(k-1)!}\\\\
f_{X}\\left(x\\right)=\\frac{(x-L)^{k-1} e^{-\\frac{x-L}{\\beta}}}{\\beta^k(k-1)!}\\\\
F^{-1}_{X}\\left(u\\right)=L+\\beta P^{-1}\\left(k,u\\right)\\\\
\\tilde{\\mu}'_{n}=E[\\tilde{X}^n]=\\int_{0}^{\\infty}x^{n}f_{\\tilde{X}}\\left(x\\right)dx=\\beta^{k}\\frac{\\Gamma\\left(n+k\\right)}{\\Gamma(k)}\\\\
\\mathrm{Mean}(X)=L+\\tilde{\\mu}'_{1}\\\\
\\mathrm{Variance}(X)=\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1}\\\\
\\mathrm{Skewness}(X)=\\frac{\\tilde{\\mu}'_{3}-3\\tilde{\\mu}'_{2}\\tilde{\\mu}'_{1}+2\\tilde{\\mu}'^{3}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{1.5}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\tilde{\\mu}'_{4}-4\\tilde{\\mu}'_{1}\\tilde{\\mu}'_{3}+6\\tilde{\\mu}'^{2}_{1}\\tilde{\\mu}'_{2}-3\\tilde{\\mu}'^{4}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{2}}\\\\
\\mathrm{Median}(X)=L+P(k,\\frac{1}{2\\beta})\\\\
\\mathrm{Mode}(X)=L+\\beta\\cdot\\left(k-1\\right)\\\\
\\tilde{X}\\sim\\mathrm{Erlang}\\left(k,\\beta\\right)\\\\
L:\\text{Location parameter}\\\\
\\beta:\\text{Scale parameter}\\\\
u:\\text{Uniform[0,1] random varible}\\\\
P\\left(a,x\\right)=\\frac{\\gamma(a,x)}{\\Gamma(a)}:\\text{Regularized lower incomplete gamma function}\\\\
P^{-1}\\left(a,u\\right):\\text{Inverse of regularized lower incomplete gamma function}\\\\
\\gamma\\left(a,x\\right):\\text{Lower incomplete gamma function}\\\\
\\Gamma\\left(x\\right):\\text{Gamma function}\\\\

ERROR_FUNCTION
X\\sim\\mathrm{ErrorFunction}\\left(h\\right)\\\\
x\\in\\left(-\\infty,\\infty\\right)\\\\
h\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=\\Phi( \\sqrt{2}hx )\\\\
f_{X}\\left(x\\right)=\\frac{h}{\\sqrt{\\pi}}e^{-h^{2}x^{2}}\\\\
F^{-1}_{X}\\left(u\\right)=\\frac{\\Phi^{-1}(u)}{\\sqrt{2}h}\\\\
\\mu'_{k}=E[X^k]=\\int_{-\\infty }^{\\infty }x^{k}f_{X}\\left(x\\right)dx\\\\
\\mathrm{Mean}(X)=\\mu'_{1}=0\\\\
\\mathrm{Variance}(X)=\\mu'_{2}-\\mu'^{2}_{1}=\\frac{1}{2h^{2}}\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}=0\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}=3\\\\
\\mathrm{Median}(X)=0\\\\
\\mathrm{Mode}(X)=0\\\\
h:\\text{Inverse of scale parameter}\\\\
u:\\text{Uniform[0,1] random varible}\\\\
\\Phi\\left(x\\right):\\text{Cumulative function from standard normal distribution}\\\\

EXPONENTIAL
X\\sim\\mathrm{Exponential}\\left(\\lambda\\right)\\\\
x\\in\\left[0,\\infty\\right)\\\\
\\lambda\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=1-e^{-\\lambda x}\\\\
f_{X}\\left(x\\right)=\\lambda e^{-\\lambda x}\\\\
F^{-1}_{X}\\left(u\\right)=-\\frac{\\ln(1-u)}{\\lambda}\\\\
\\mu'_{k}=E[X^k]=\\int_{0}^{\\infty}x^{n}f_{X}\\left(x\\right)dx=\\frac{k!}{\\lambda^{k}}\\\\
\\mathrm{Mean}(X)=\\mu'_{1}=\\frac{1}{\\lambda}\\\\
\\mathrm{Variance}(X)=\\mu'_{2}-\\mu'^{2}_{1}=\\frac{1}{\\lambda^2}\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}=2\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}=9\\\\
\\mathrm{Median}(X)=\\frac{\\ln 2}{\\lambda}\\\\
\\mathrm{Mode}(X)=0\\\\
\\lambda:\\text{Inverse of scale parameter}\\\\
u:\\text{Uniform[0,1] random varible}\\\\

EXPONENTIAL_2P
X\\sim\\mathrm{Exponential_{2P}}\\left(\\lambda,L\\right)\\\\
x\\in\\left[L,\\infty\\right)\\\\
\\lambda\\in\\mathbb{R}^{+},L\\in\\mathbb{R}\\\\
F_{X}\\left(x\\right)=1-e^{-\\lambda (x-L)}\\\\
f_{X}\\left(x\\right)=\\lambda e^{-\\lambda (x-L)}\\\\
F^{-1}_{X}\\left(u\\right)=L-\\frac{\\ln(1-u)}{\\lambda}\\\\
\\tilde{\\mu}'_{k}=E[\\tilde{X}^k]=\\int_{0}^{\\infty}x^{k}f_{\\tilde{X}}\\left(x\\right)dx=\\frac{k!}{\\lambda^{k}}\\\\
\\mathrm{Mean}(X)=L+\\tilde{\\mu}'_{1}=L+\\frac{1}{\\lambda}\\\\
\\mathrm{Variance}(X)=\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1}=\\frac{1}{\\lambda^2}\\\\
\\mathrm{Skewness}(X)=\\frac{\\tilde{\\mu}'_{3}-3\\tilde{\\mu}'_{2}\\tilde{\\mu}'_{1}+2\\tilde{\\mu}'^{3}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{1.5}}=2\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\tilde{\\mu}'_{4}-4\\tilde{\\mu}'_{1}\\tilde{\\mu}'_{3}+6\\tilde{\\mu}'^{2}_{1}\\tilde{\\mu}'_{2}-3\\tilde{\\mu}'^{4}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{2}}=9\\\\
\\mathrm{Median}(X)=L+\\frac{\\ln 2}{\\lambda}\\\\
\\mathrm{Mode}(X)=L\\\\
\\tilde{X}\\sim\\mathrm{Exponential}\\left(\\lambda\\right)\\\\
L:\\text{Location parameter}\\\\
\\lambda:\\text{Inverse of scale parameter}\\\\
u:\\text{Uniform[0,1] random varible}\\\\

F
X\\sim\\mathrm{F}\\left(df_1,df_2\\right)\\\\
x\\in\\left[0,\\infty\\right)\\\\
df_{1}\\in\\mathbb{R}^{+},df_{2}\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=I_{df_1 x/(df_1 x+df_2)}\\left (\\tfrac{df_1}{2},\\tfrac{df_2}{2}\\right)\\\\
f_{X}\\left(x\\right)=\\frac{\\sqrt{\\frac{(df_1 x)^{df_1} df_2^{df_2}}{(df_1 x+df_2)^{df_1+df_2}}}}{x\\,Beta\\left(\\frac{df_1}{2},\\frac{df_2}{2}\\right)}\\\\
F^{-1}_{X}\\left(u\\right)=\\frac{df_2\\times I^{-1}\\left(u,\\frac{df_1}{2},\\frac{df_2}{2}\\right)}{df_1\\times \\left(1-I^{-1}\\left(u,\\frac{df_1}{2},\\frac{df_2}{2}\\right)\\right)}\\\\
\\mu'_{k}=E[X^k]=\\int_{0}^{\\infty}x^{k }f_{X}\\left(x\\right)dx=\\left(\\frac{df_2}{df_1}\\right)^k\\frac{\\Gamma\\left(\\tfrac{df_1}{2}+k\\right) }{\\Gamma\\left(\\tfrac{df_1}{2}\\right)}\\frac{\\Gamma\\left(\\tfrac{df_2}{2}-k\\right) }{\\Gamma\\left(\\tfrac{df_2}{2}\\right) } \\quad \\text{if }df_2\\gt 2k\\\\
\\mathrm{Mean}(X)=\\mu'_{1}=\\frac{df_2}{df_2-2} \\quad \\text{if }df_2\\gt 2\\\\
\\mathrm{Variance}(X)=\\mu'_{2}-\\mu'^{2}_{1}=\\frac{2\\,df_2^2\\,(df_1+df_2-2)}{df_1 (df_2-2)^2 (df_2-4)} \\quad \\text{if }df_2\\gt 4\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}=\\frac{(2 df_1+df_2-2) \\sqrt{8 (df_2-4)}}{(df_2-6) \\sqrt{df_1 (df_1+df_2 -2)}}\\quad \\text{if }df_2\\gt 6\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}=\\frac{3\\left(8+\\left(df_2-6\\right)\\times \\mathrm{Skewness}(X)^{2}\\right)}{2df_2-16}+3\\quad \\text{if }df_2\\gt 8\\\\
\\mathrm{Median}(X)=\\frac{df_2\\times I^{-1}\\left(\\frac{1}{2},\\frac{df_1}{2},\\frac{df_2}{2}\\right)}{df_1\\times \\left(1-I^{-1}\\left(\\frac{1}{2},\\frac{df_1}{2},\\frac{df_2}{2}\\right)\\right)}\\\\
\\mathrm{Mode}(X)=\\frac{df_2\\left(df_1-2\\right)}{df_1\\left(df_2+2\\right)}  \\quad \\text{if }df_1\\gt 2\\\\
u:\\text{Uniform[0,1] random varible}\\\\
I\\left(x,a,b\\right):\\text{Regularized incomplete beta function}\\\\
I^{-1}\\left(x,a,b\\right):\\text{Inverse of regularized incomplete beta function}\\\\
Beta\\left(x,y\\right):\\text{Beta function}\\\\

F_4P
X\\sim\\mathrm{F_{4P}}\\left(df_1,df_2,L,S\\right)\\\\
x\\in\\left[L,\\infty\\right)\\\\
df_{1}\\in\\mathbb{R}^{+},df_{2}\\in\\mathbb{R}^{+},L\\in\\mathbb{R},S\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=I_{df_1 z(x)/(df_1 z(x)+df_2)}\\left (\\tfrac{df_1}{2},\\tfrac{df_2}{2}\\right)\\\\
f_{X}\\left(x\\right)=\\frac{1}{S}\\times \\frac{\\sqrt{\\frac{(df_1 z(x))^{df_1} df_2^{df_2}}{(df_1 z(x)+df_2)^{df_1+df_2}}}}{z(x)\\,Beta\\left(\\frac{df_1}{2},\\frac{df_2}{2}\\right)}\\\\
F^{-1}_{X}\\left(u\\right)=L+S\\frac{df_2\\times I^{-1}\\left(u,\\frac{df_1}{2},\\frac{df_2}{2}\\right)}{df_1\\times \\left(1-I^{-1}\\left(u,\\frac{df_1}{2},\\frac{df_2}{2}\\right)\\right)}\\\\
\\tilde{\\mu}'_{k}=E[\\tilde{X}^k]=\\int_{0}^{\\infty}x^{k}f_{\\tilde{X}}\\left(x\\right)dx=\\frac{\\Gamma\\left(\\tfrac{df_1}{2}+k\\right) }{\\Gamma\\left(\\tfrac{df_1}{2}\\right)}\\frac{\\Gamma\\left(\\tfrac{df_2}{2}-k\\right) }{\\Gamma\\left(\\tfrac{df_2}{2}\\right) }\\left(\\frac{df_2}{df_1}\\right)^k \\quad \\text{if }df_2\\gt 2k\\\\
\\mathrm{Mean}(X)=L+S\\tilde{\\mu}'_{1}=L+S\\frac{df_2}{df_2-2} \\quad \\text{if }df_2\\gt 2\\\\
\\mathrm{Variance}(X)=S^{2}(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})=S^{2}\\frac{2\\,df_2^2\\,(df_1+df_2-2)}{df_1 (df_2-2)^2 (df_2-4)} \\quad \\text{if }df_2\\gt 4\\\\
\\mathrm{Skewness}(X)=\\frac{\\tilde{\\mu}'_{3}-3\\tilde{\\mu}'_{2}\\tilde{\\mu}'_{1}+2\\tilde{\\mu}'^{3}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{1.5}}=\\frac{(2 df_1+df_2-2) \\sqrt{8 (df_2-4)}}{(df_2-6) \\sqrt{df_1 (df_1+df_2 -2)}}\\quad \\text{if }df_2\\gt 6\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\tilde{\\mu}'_{4}-4\\tilde{\\mu}'_{1}\\tilde{\\mu}'_{3}+6\\tilde{\\mu}'^{2}_{1}\\tilde{\\mu}'_{2}-3\\tilde{\\mu}'^{4}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{2}}=\\frac{3\\left(8+\\left(df_2-6\\right)\\times \\mathrm{Skewness}(X)^{2}\\right)}{2df_2-16}+3\\quad \\text{if }df_2\\gt 8\\\\
\\mathrm{Median}(X)=L+S\\frac{df_2\\times I^{-1}\\left(\\frac{1}{2},\\frac{df_1}{2},\\frac{df_2}{2}\\right)}{df_1\\times \\left(1-I^{-1}\\left(\\frac{1}{2},\\frac{df_1}{2},\\frac{df_2}{2}\\right)\\right)}\\\\
\\mathrm{Mode}(X)=L+S\\frac{df_2\\left(df_1-2\\right)}{df_1\\left(df_2+2\\right)}  \\quad \\text{if }df_1\\gt 2\\\\
\\tilde{X}\\sim\\mathrm{F}\\left(df_1,df_2\\right)\\\\
L:\\text{Location parameter}\\\\
S:\\text{Scale parameter}\\\\
z\\left(x\\right)=\\left(x-L\\right)/S\\\\
u:\\text{Uniform[0,1] random varible}\\\\
I\\left(x,a,b\\right):\\text{Regularized incomplete beta function}\\\\
I^{-1}\\left(x,a,b\\right):\\text{Inverse of regularized incomplete beta function}\\\\
Beta\\left(x,y\\right):\\text{Beta function}\\\\

FATIGUE_LIFE
X\\sim\\mathrm{FatigueLife}\\left(\\gamma,L,S\\right)\\\\
x\\in\\left(L,\\infty\\right)\\\\
\\gamma\\in\\mathbb{R}^{+},L\\in\\mathbb{R},S\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=\\Phi\\left(\\frac{\\sqrt{z(x)}-\\sqrt{\\frac{1}{z(x)}}}{\\gamma}\\right)\\\\
f_{X}\\left(x\\right)=\\frac{\\sqrt{z(x)}+\\sqrt{\\frac{1}{z(x)}}}{2\\gamma z(x)}\\phi\\left(\\frac{\\sqrt{z(x)}-\\sqrt{\\frac{1}{z(x)}}}{\\gamma}\\right)\\\\
F^{-1}_{X}\\left(u\\right)=L+S\\frac{1}{4}\\left[\\gamma\\Phi^{-1}(u)+\\sqrt{4+\\left(\\gamma\\Phi^{-1}(u)\\right)^2}\\right]^2\\\\
\\mu'_{k}=E[X^k]=\\int_{-\\infty }^{\\infty }x^{k}f_{X}\\left(x\\right)dx\\\\
\\mathrm{Mean}(X)=L+S\\mu'_{1}=L+S\\left(1+\\frac{\\gamma^2}{2}\\right)\\\\
\\mathrm{Variance}(X)=S^{2}(\\mu'_{2}-\\mu'^{2}_{1})=S^{2}\\gamma^{2}\\left(1+\\frac{5\\gamma^{2}}{4}\\right)\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}=\\frac{4\\gamma(6+11\\gamma^2)}{(4+5\\gamma^2)^{1.5}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}=3+\\frac{ 6 \\gamma^2 ( 93 \\gamma^2+40 ) }{ ( 5 \\gamma^2+4 )^2 }\\\\
\\mathrm{Median}(X)=L+S\\frac{1}{4}\\left[\\gamma\\Phi^{-1}\\left(1/2\\right)+\\sqrt{4+\\left(\\gamma\\Phi^{-1}\\left(1/2\\right)\\right)^2}\\right]^2\\\\
\\mathrm{Mode}(X)=\\arg\\max_{x}f_{X}\\left(x\\right)\\\\
L:\\text{Location parameter}\\\\
S:\\text{Scale parameter}\\\\
z\\left(x\\right)=\\left(x-L\\right)/S\\\\
u:\\text{Uniform[0,1] random varible}\\\\
\\Phi\\left(x\\right):\\text{Cumulative function from standard normal distribution}\\\\
\\phi\\left(x\\right):\\text{Density function from standard normal distribution}\\\\

FOLDED_NORMAL
X\\sim\\mathrm{FoldedNormal}\\left(\\mu,\\sigma\\right)\\\\
x\\in\\left[0,\\infty\\right)\\\\
\\mu\\in\\mathbb{R},\\sigma\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=\\frac{1}{2}\\left[\\text{erf}\\left(\\frac{x+\\mu}{\\sigma\\sqrt{2}}\\right)+\\text{erf}\\left(\\frac{x-\\mu}{\\sigma\\sqrt{2}}\\right)\\right]\\\\
f_{X}\\left(x\\right)=\\frac{1}{\\sigma\\sqrt{2\\pi}} \\,e^{ -\\frac{(x-\\mu)^2}{2\\sigma^2} }+\\frac{1}{\\sigma\\sqrt{2\\pi}} \\,e^{ -\\frac{(x+\\mu)^2}{2\\sigma^2} }\\\\
F^{-1}_{X}\\left(u\\right)=\\left|\\mu+\\sigma\\Phi^{-1}(u)\\right|\\\\
\\mu'_{k}=E[X^k]=\\int_{0}^{\\infty }x^{k}f_{X}\\left(x\\right)dx\\\\
\\mathrm{Mean}(X)=\\mu'_{1}=\\sigma \\sqrt{\\tfrac{2}{\\pi}} \\,e^{(-\\mu^2/2\\sigma^2)}+\\mu\\left(1-2\\,\\Phi(-\\tfrac{\\mu}{\\sigma})\\right)\\\\
\\mathrm{Variance}(X)=\\mu'_{2}-\\mu'^{2}_{1}=\\mu^2+\\sigma^2-\\mathrm{Mean}(X)^{2}\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}\\\\
\\mathrm{Median}(X)=\\left|\\mu+\\sigma\\Phi^{-1}\\left(1/2\\right)\\right|\\\\
\\mathrm{Mode}(X)=\\arg\\max_{x}f_{X}\\left(x\\right)\\\\
\\mu:\\text{Location parameter}\\\\
\\sigma:\\text{Scale parameter}\\\\
u:\\text{Uniform[0,1] random varible}\\\\
\\Phi\\left(x\\right):\\text{Cumulative function from standard normal distribution}\\\\
\\phi\\left(x\\right):\\text{Density function from standard normal distribution}\\\\
\\Phi^{-1}\\left(x\\right):\\text{Invere of cumulative from function standard normal distribution}\\\\

FRECHET
X\\sim\\mathrm{Frechet}\\left(\\alpha,L,S\\right)\\\\
x\\in\\left[L,\\infty\\right)\\\\
\\alpha\\in\\mathbb{R}^{+},L\\in\\mathbb{R},S\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=e^{(-z(x))^{-\\alpha}}\\\\
f_{X}\\left(x\\right)=\\frac{\\alpha}{S} \\,\\left(z(x)\\right)^{-1-\\alpha} \\,e^{-(z(x))^{-\\alpha}}\\\\
F^{-1}_{X}\\left(u\\right)=L+S\\left(-\\ln(u)\\right)^{-\\frac{1}{\\alpha}}\\\\
\\mu'_{k}=E[X^k]=\\int_{L}^{\\infty }x^{k}f_{X}\\left(x\\right)dx=\\Gamma\\left(1-\\frac{k}{\\alpha}\\right)\\\\
\\mathrm{Mean}(X)=L+S\\mu'_{1} \\quad \\text{if } \\alpha>1\\\\
\\mathrm{Variance}(X)=S^{2}(\\mu'_{2}-\\mu'^{2}_{1}) \\quad \\text{if } \\alpha>2\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}} \\quad \\text{if } \\alpha>3\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}} \\quad \\text{if } \\alpha>4\\\\
\\mathrm{Median}(X)=L+\\frac{S}{\\sqrt[\\alpha]{\\ln(2)}}\\\\
\\mathrm{Mode}(X)=L+S\\left(\\frac{\\alpha}{1+\\alpha}\\right)^{1/\\alpha}\\\\
L:\\text{Location parameter}\\\\
S:\\text{Scale parameter}\\\\
z\\left(x\\right)=\\left(x-L\\right)/S\\\\
u:\\text{Uniform[0,1] random varible}\\\\
\\Gamma\\left(x\\right):\\text{Gamma function}\\\\

GAMMA
X\\sim\\mathrm{Gamma}\\left(\\alpha,\\beta\\right)\\\\
x\\in\\left(0,\\infty\\right)\\\\
\\alpha\\in\\mathbb{R}^{+},\\beta\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=P\\left(\\alpha,\\frac{x}{\\beta}\\right)=\\frac{1}{\\Gamma(\\alpha)} \\gamma\\left(\\alpha,\\frac{x}{\\beta}\\right)\\\\
f_{X}\\left(x\\right)=\\frac{1}{\\Gamma(\\alpha) \\beta^\\alpha} x^{\\alpha-1} e^{-\\frac{x}{\\beta}}\\\\
F^{-1}_{X}\\left(u\\right)=\\beta P^{-1}\\left(\\alpha,u\\right)\\\\
\\mu'_{k}=E[X^k]=\\int_{0}^{\\infty }x^{k}f_{X}\\left(x\\right)dx=\\beta^k\\frac{\\Gamma(k+\\alpha)}{\\Gamma(\\alpha)}\\\\
\\mathrm{Mean}(X)=\\mu'_{1}=\\alpha \\beta\\\\
\\mathrm{Variance}(X)=\\mu'_{2}-\\mu'^{2}_{1}=\\alpha \\beta^2\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}=\\frac{2}{\\sqrt{\\alpha}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}=3+\\frac{6}{\\alpha}\\\\
\\mathrm{Median}(X)=(\\alpha-1)\\beta \\quad \\text{if }\\alpha>1\\\\
\\mathrm{Mode}(X)=\\beta P^{-1}\\left(\\alpha,\\frac{1}{2}\\right)\\\\
\\beta:\\text{Scale parameter}\\\\
u:\\text{Uniform[0,1] random varible}\\\\
P\\left(a,x\\right)=\\frac{\\gamma(a,x)}{\\Gamma(a)}:\\text{Regularized lower incomplete gamma function}\\\\
P^{-1}\\left(a,u\\right):\\text{Inverse of regularized lower incomplete gamma function}\\\\
\\gamma\\left(a,x\\right):\\text{Lower incomplete gamma function}\\\\
\\Gamma\\left(x\\right):\\text{Gamma function}\\\\

GAMMA_3P
X\\sim\\mathrm{Gamma_{3P}}\\left(\\alpha,\\beta,L\\right)\\\\
x\\in\\left(L,\\infty\\right)\\\\
\\alpha\\in\\mathbb{R}^{+},\\beta\\in\\mathbb{R}^{+},L\\in\\mathbb{R}\\\\
F_{X}\\left(x\\right)=P\\left(\\alpha,\\frac{x-L}{\\beta}\\right)=\\frac{1}{\\Gamma(\\alpha)} \\gamma\\left(\\alpha,\\frac{x-L}{\\beta}\\right)\\\\
f_{X}\\left(x\\right)=\\frac{1}{\\Gamma(\\alpha) \\beta^\\alpha} (x-L)^{\\alpha-1} e^{-\\frac{x-L}{\\beta}}\\\\
F^{-1}_{X}\\left(u\\right)=L+\\beta P^{-1}\\left(\\alpha,u\\right)\\\\
\\tilde{\\mu}'_{k}=E[\\tilde{X}^k]=\\int_{0}^{\\infty}x^{k}f_{\\tilde{X}}\\left(x\\right)dx=\\beta^k\\frac{\\Gamma(k+\\alpha)}{\\Gamma(\\alpha)}\\\\
\\mathrm{Mean}(X)=L+\\tilde{\\mu}'_{1}=L+\\alpha \\beta\\\\
\\mathrm{Variance}(X)=\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1}=\\alpha \\beta^2\\\\
\\mathrm{Skewness}(X)=\\frac{\\tilde{\\mu}'_{3}-3\\tilde{\\mu}'_{2}\\tilde{\\mu}'_{1}+2\\tilde{\\mu}'^{3}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{1.5}}=\\frac{2}{\\sqrt{\\alpha}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\tilde{\\mu}'_{4}-4\\tilde{\\mu}'_{1}\\tilde{\\mu}'_{3}+6\\tilde{\\mu}'^{2}_{1}\\tilde{\\mu}'_{2}-3\\tilde{\\mu}'^{4}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{2}}=3+\\frac{6}{\\alpha}\\\\
\\mathrm{Median}(X)=L+(\\alpha-1)\\beta \\quad \\text{if }\\alpha>1\\\\
\\mathrm{Mode}(X)=L+\\beta P^{-1}\\left(\\alpha,\\frac{1}{2}\\right)\\\\
\\tilde{X}\\sim\\mathrm{Gamma}\\left(\\alpha,\\beta\\right)\\\\
\\beta:\\text{Scale parameter}\\\\
L:\\text{Location parameter}\\\\
u:\\text{Uniform[0,1] random varible}\\\\
P\\left(a,x\\right)=\\frac{\\gamma(a,x)}{\\Gamma(a)}:\\text{Regularized lower incomplete gamma function}\\\\
P^{-1}\\left(a,u\\right):\\text{Inverse of regularized lower incomplete gamma function}\\\\
\\gamma\\left(a,x\\right):\\text{Lower incomplete gamma function}\\\\
\\Gamma\\left(x\\right):\\text{Gamma function}\\\\

GENERALIZED_EXTREME_VALUE
X\\sim\\mathrm{GeneralizedExtremeValue}\\left(\\xi,\\mu,\\sigma\\right)\\\\
\\text{if }\\xi>0:\\ x\\in\\left(z(x),\\infty\\right),\\quad \\text{if }\\xi=0:\\ x\\in\\left(-\\infty,\\infty\\right),\\quad \\text{if }\\xi<0:\\ x\\in\\left(-\\infty,z(x)\\right)\\\\
\\xi\\in\\mathbb{R},\\mu\\in\\mathbb{R},\\sigma\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=\\left\\{\\begin{array}{cl} \\exp\\Bigl(-\\exp(-z(x))\\Bigr) & \\text{if } ~ \\xi=0 \\\\ \\exp\\Bigl(-(1+\\xi z(x))^{-1/\\xi}\\Bigr) & \\text{if } ~ \\xi \\neq 0\\end{array} \\right.\\\\
f_{X}\\left(x\\right)=\\left\\{\\begin{array}{cl}\\frac{1}{\\sigma}\\exp(-z(x)) \\exp\\Bigl(-\\exp(-z(x))\\Bigr) & \\text{if } ~ \\xi=0 \\\\ \\frac{1}{\\sigma}\\Bigl(1+\\xi z(x)\\Bigr)^{-(1+1/\\xi)} \\exp\\Bigl(-(1+\\xi z(x))^{-1/\\xi}\\Bigr) & \\text{if } ~ \\xi \\neq 0\\end{array} \\right.\\\\
F^{-1}_{X}\\left(u\\right)=\\left\\{\\begin{array}{cl} \\mu-\\sigma\\ln\\left(-\\ln\\left(u\\right)\\right) & \\text{if } ~ \\xi=0 \\\\ \\mu+\\displaystyle{{\\,\\sigma\\,}\\over{\\,\\xi\\,}}\\left(\\left(-\\ln(u)\\,\\right)^{-\\xi}-1\\right) & \\text{if } ~ \\xi \\neq 0\\\\ \\end{array} \\right.\\\\
\\mu'_{k}=E[X^k]=\\int_{-\\infty}^{\\infty}x^{k}f_{X}\\left(x\\right)dx=\\Gamma(1-k\\xi)\\\\
\\mathrm{Mean}(X)=\\left\\{\\begin{array}{cl}\\mu+\\sigma(\\mu'_{1}-1)/\\xi & \\text{if}\\ \\xi\\neq 0,\\xi<1\\\\ \\mu+\\sigma\\,\\gamma & \\text{if}\\ \\xi=0\\end{array} \\right.\\\\
\\mathrm{Variance}(X)=\\left\\{\\begin{array}{cl}\\sigma^2\\,(\\mu'_{2}-\\mu'^{2}_{1})/\\xi^2 & \\text{if}\\ \\xi\\neq0,\\xi<\\frac12\\\\ \\sigma^2\\,\\frac{\\pi^2}{6} & \\text{if}\\ \\xi=0\\end{array} \\right.\\\\
\\mathrm{Skewness}(X)=\\left\\{\\begin{array}{cl}\\text{sign}(\\xi)\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}} & \\text{if}\\ \\xi\\neq0,\\xi<\\frac{1}{3} \\\\ \\frac{12 \\sqrt{6}\\,\\zeta(3)}{\\pi^3} & \\text{if}\\ \\xi=0\\end{array} \\right.\\\\
\\mathrm{Kurtosis}(X)=\\left\\{\\begin{array}{cl}3+\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}} & \\text{if}\\ \\xi\\neq0,\\xi<\\frac{1}{4}\\\\ 3+\\frac{12}{5} & \\text{if}\\ \\xi=0\\end{array} \\right.\\\\
\\mathrm{Median}(X)=\\left\\{\\begin{array}{cl}\\mu+\\sigma\\frac{(\\ln2)^{-\\xi}-1}{\\xi} & \\text{if }\\ \\xi\\neq0\\\\ \\mu-\\sigma \\ln\\ln2 & \\text{if}\\ \\xi=0\\end{array} \\right.\\\\
\\mathrm{Mode}(X)=\\left\\{\\begin{array}{cl}\\mu+\\sigma\\frac{(1+\\xi)^{-\\xi}-1}{\\xi} & \\text{if }\\ \\xi\\neq0\\\\ \\mu & \\text{if }\\ \\xi=0\\end{array} \\right.\\\\
\\mu:\\text{Location parameter}\\\\
\\sigma:\\text{Scale parameter}\\\\
z\\left(x\\right)=\\left(x-\\mu\\right)/\\sigma\\\\
u:\\text{Uniform[0,1] random varible}\\\\
\\Gamma\\left(x\\right):\\text{Gamma function}\\\\
\\gamma:\\text{Euler–Mascheroni constant}=0.5772156649\\\\

GENERALIZED_GAMMA
X\\sim\\mathrm{GeneralizedGamma}\\left(a,d,p\\right)\\\\
x\\in\\left(0,\\infty\\right)\\\\
a\\in\\mathbb{R}^{+},d\\in\\mathbb{R}^{+},p\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=P(d/p,(x/a)^p)=\\frac{\\gamma(d/p,(x/a)^p)}{\\Gamma(d/p)}\\\\
f_{X}\\left(x\\right)=\\frac{p/a^d}{\\Gamma(d/p)} x^{d-1}e^{-(x/a)^p}\\\\
F^{-1}_{X}\\left(u\\right)=aP^{-1}\\left(\\frac{d}{p},u\\right)^{\\frac{1}{p}}\\\\
\\mu'_{k}=E[X^k]=\\int_{0}^{\\infty}x^{k}f_{X}\\left(x\\right)dx=a^k\\frac{\\Gamma (\\frac{d+k}{p})}{\\Gamma(\\frac{d}{p})}\\\\
\\mathrm{Mean}(X)=\\mu'_{1}\\\\
\\mathrm{Variance}(X)=\\mu'_{2}-\\mu'^{2}_{1}\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}\\\\
\\mathrm{Median}(X)=aP^{-1}\\left(\\frac{d}{p},\\frac{1}{2}\\right)^{\\frac{1}{p}}\\\\
\\mathrm{Mode}(X)=a\\left(\\frac{d-1}{p}\\right)^{\\frac{1}{p}} \\quad \\text{if } d>1\\\\
a:\\text{Scale parameter}\\\\
u:\\text{Uniform[0,1] random varible}\\\\
P\\left(a,x\\right)=\\frac{\\gamma(a,x)}{\\Gamma(a)}:\\text{Regularized lower incomplete gamma function}\\\\
P^{-1}\\left(a,u\\right):\\text{Inverse of regularized lower incomplete gamma function}\\\\
\\gamma\\left(a,x\\right):\\text{Lower incomplete gamma function}\\\\
\\Gamma\\left(x\\right):\\text{Gamma function}\\\\

GENERALIZED_GAMMA_4P
X\\sim\\mathrm{GeneralizedGamma_{4P}}\\left(a,d,p,L\\right)\\\\
x\\in\\left(L,\\infty\\right)\\\\
a\\in\\mathbb{R}^{+},d\\in\\mathbb{R}^{+},p\\in\\mathbb{R}^{+},L\\in\\mathbb{R}\\\\
F_{X}\\left(x\\right)=P(d/p,((x-L)/a)^p)=\\frac{\\gamma(d/p,((x-L)/a)^p)}{\\Gamma(d/p)}\\\\
f_{X}\\left(x\\right)=\\frac{p/a^d}{\\Gamma(d/p)} (x-L)^{d-1}e^{-((x-L)/a)^p}\\\\
F^{-1}_{X}\\left(u\\right)=L+aP^{-1}\\left(\\frac{d}{p},u\\right)^{\\frac{1}{p}}\\\\
\\tilde{\\mu}'_{k}=E[\\tilde{X}^k]=\\int_{0}^{\\infty}x^{k}f_{\\tilde{X}}\\left(x\\right)dx=a^k\\frac{\\Gamma (\\frac{d+k}{p})}{\\Gamma(\\frac{d}{p})}\\\\
\\mathrm{Mean}(X)=L+\\tilde{\\mu}'_{1}\\\\
\\mathrm{Variance}(X)=\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1}\\\\
\\mathrm{Skewness}(X)=\\frac{\\tilde{\\mu}'_{3}-3\\tilde{\\mu}'_{2}\\tilde{\\mu}'_{1}+2\\tilde{\\mu}'^{3}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{1.5}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\tilde{\\mu}'_{4}-4\\tilde{\\mu}'_{1}\\tilde{\\mu}'_{3}+6\\tilde{\\mu}'^{2}_{1}\\tilde{\\mu}'_{2}-3\\tilde{\\mu}'^{4}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{2}}\\\\
\\mathrm{Median}(X)=L+aP^{-1}\\left(\\frac{d}{p},\\frac{1}{2}\\right)^{\\frac{1}{p}}\\\\
\\mathrm{Mode}(X)=L+a\\left(\\frac{d-1}{p}\\right)^{\\frac{1}{p}} \\quad \\text{if } d>1\\\\
\\tilde{X}\\sim\\mathrm{GeneralizedGamma_{4P}}\\left(a,d,p\\right)\\\\
L:\\text{Location parameter}\\\\
a:\\text{Scale parameter}\\\\
u:\\text{Uniform[0,1] random varible}\\\\
P\\left(a,x\\right)=\\frac{\\gamma(a,x)}{\\Gamma(a)}:\\text{Regularized lower incomplete gamma function}\\\\
P^{-1}\\left(a,u\\right):\\text{Inverse of regularized lower incomplete gamma function}\\\\
\\gamma\\left(a,x\\right):\\text{Lower incomplete gamma function}\\\\
\\Gamma\\left(x\\right):\\text{Gamma function}\\\\

GENERALIZED_LOGISTIC
X\\sim\\mathrm{GeneralizedLogistic}\\left(c,L,S\\right)\\\\
x\\in\\left(L,\\infty\\right)\\\\
c\\in\\mathbb{R}^{+},L\\in\\mathbb{R},S\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=\\frac{1}{\\left(1+\\exp\\left(-z(x)\\right)\\right)^{c}}\\\\
f_{X}\\left(x\\right)=\\frac{c\\exp\\left(-z(x)\\right)}{S\\left(1+\\exp\\left(-z(x)\\right)\\right)^{c+1}}\\\\
F^{-1}_{X}\\left(u\\right)=L-S\\ln\\left(u^{-1/c}-1\\right)\\\\
\\mu'_{k}=E[X^k]=\\int_{-\\infty }^{\\infty }x^{k}f_{X}\\left(x\\right)dx\\\\
\\mathrm{Mean}(X)=L+S\\mu'_{1}=L+S\\left(\\gamma+\\psi_{0}\\left(c\\right)\\right)\\\\
\\mathrm{Variance}(X)=S^{2}(\\mu'_{2}-\\mu'^{2}_{1})=S^{2}\\left(\\frac{\\pi^{2}}{6}+\\psi_{1}\\left(c\\right)\\right)\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}=\\frac{\\psi_{2}\\left(c\\right)+2\\zeta\\left(3\\right)}{\\left(\\frac{\\pi^{2}}{6}+\\psi_{1}\\left(c\\right)\\right)^{3/2}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}=\\frac{\\left(\\frac{\\pi^{4}}{15}+\\psi_{3}\\left(c\\right)\\right)}{\\left(\\frac{\\pi^{2}}{6}+\\psi_{1}\\left(c\\right)\\right)^{2}}\\\\
\\mathrm{Median}(X)=L-S\\ln\\left(2^{1/c}-1\\right)\\\\
\\mathrm{Mode}(X)=L+S\\ln\\left(c\\right)\\\\
L:\\text{Location parameter}\\\\
S:\\text{Scale parameter}\\\\
z\\left(x\\right)=\\left(x-L\\right)/S\\\\
u:\\text{Uniform[0,1] random varible}\\\\
\\gamma:\\text{Euler–Mascheroni constant}=0.5772156649\\\\
\\psi_{0}\\left(x\\right):\\text{Digamma function}\\\\
\\psi_{n}\\left(x\\right):\\text{Polygamma function of order }n\\in\\mathbb{N}\\\\

GENERALIZED_NORMAL
X\\sim\\mathrm{GeneralizedNormal}\\left(\\mu,\\alpha,\\beta\\right)\\\\
x\\in\\left(-\\infty,+\\infty\\right)\\\\
\\mu\\in\\mathbb{R},\\alpha\\in\\mathbb{R}^{+},\\beta\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=\\frac{1}{2}+\\frac{\\text{sign}(x-\\mu)}{2\\Gamma( 1/\\beta ) } \\gamma\\left(1/\\beta,\\left|\\frac{x-\\mu}{\\alpha}\\right|^\\beta\\right)=\\frac{1}{2}+\\frac{\\text{sign}(x-\\mu)}{2} P\\left(1/\\beta,\\left|\\frac{x-\\mu}{\\alpha}\\right|^\\beta\\right)\\\\
f_{X}\\left(x\\right)=\\frac{\\beta}{2\\alpha\\Gamma(1/\\beta)}\\exp\\left(-\\left(\\frac{|x-\\mu|}{\\alpha}\\right)^\\beta\\right)\\\\
F^{-1}_{X}\\left(u\\right)=\\text{sign}(p-frac{1}{2}) \\left[\\alpha^\\beta P^{-1}\\left(\\frac{1}{\\beta},2|u-frac{1}{2}|\\right)\\right]^{1/\\beta}+\\mu\\\\
\\mu'_{k}=E[X^k]=\\int_{-\\infty}^{\\infty}x^{k}f_{X}\\left(x\\right)dx=\\left\\{\\begin{array}{cl}0 & \\text{if }k\\text{ is odd} \\\\ \\alpha^{k} \\Gamma\\left(\\frac{k+1}{\\beta}\\right) \\Big/ \\Gamma\\left(\\frac{1}{\\beta}\\right) & \\text{if }k\\text{ is even}\\end{array} \\right.\\\\
\\mathrm{Mean}(X)=\\mu+\\alpha\\mu'_{1}=\\mu\\\\
\\mathrm{Variance}(X)=\\alpha^2(\\mu'_{2}-\\mu'^{2}_{1})=\\frac{\\alpha^2\\Gamma(3/\\beta)}{\\Gamma(1/\\beta)}\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}=0\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}=\\frac{\\Gamma(5/\\beta)\\Gamma(1/\\beta)}{\\Gamma(3/\\beta)^2}\\\\
\\mathrm{Median}(X)=\\mu\\\\
\\mathrm{Mode}(X)=\\mu\\\\
\\mu:\\text{Location parameter}\\\\
\\alpha:\\text{Scale parameter}\\\\
u:\\text{Uniform[0,1] random varible}\\\\
P^{-1}\\left(a,u\\right):\\text{Inverse of regularized lower incomplete gamma function}\\\\
\\gamma\\left(a,x\\right):\\text{Lower incomplete gamma function}\\\\
\\Gamma\\left(x\\right):\\text{Gamma function}\\\\

GENERALIZED_PARETO
X\\sim\\mathrm{GeneralizedPareto}\\left(c,\\mu,\\sigma\\right)\\\\
\\text{if }c\\geqslant 0:\\ x\\in\\left(\\mu,\\infty\\right),\\quad \\text{if }c<0:\\ x\\in\\left(-\\infty,\\mu-\\frac{\\sigma}{c}\\right)\\\\
c\\in\\mathbb{R},\\mu\\in\\mathbb{R},\\sigma\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=1-(1+c z(x))^{-1/c}\\\\
f_{X}\\left(x\\right)=\\frac{1}{\\sigma}(1+c z(x))^{-(1/c +1)}\\\\
F^{-1}_{X}\\left(u\\right)=\\mu+\\frac{\\sigma (u^{-c}-1)}{c}\\\\
\\mu'_{k}=E[X^k]=\\int_{-\\infty}^{\\infty}x^{k}f_{X}\\left(x\\right)dx=\\frac{\\left(-1\\right)^{k}}{c^{k}}\\sum_{i=0}^{k}\\binom{k}{i}\\frac{\\left(-1\\right)^{i}}{1-ci}\\quad \\text{if }<\\frac{1}{k}\\\\
\\mathrm{Mean}(X)=\\mu+\\sigma\\mu'_{1}=\\mu+\\frac{\\sigma}{1-c} \\quad \\text{if } c<1\\\\
\\mathrm{Variance}(X)=\\sigma^2(\\mu'_{2}-\\mu'^{2}_{1})=\\frac{\\sigma^2}{(1-c)^2(1-2c)} \\quad \\text{if } c<1/2\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}=\\frac{2(1+c)\\sqrt{1-2c}}{(1-3c)} \\quad \\text{if } c<1/3\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}=\\frac{3(1-2c)(2c^2+c+3)}{(1-3c)(1-4c)} \\quad \\text{if } c<1/4\\\\
\\mathrm{Median}(X)=\\mu\\\\
\\mathrm{Mode}(X)=\\mu+\\frac{\\sigma( 2^{c} -1)}{c}\\\\
\\mu:\\text{Location parameter}\\\\
\\sigma:\\text{Scale parameter}\\\\
z\\left(x\\right)=\\left(x-\\mu\\right)/\\sigma\\\\
u:\\text{Uniform[0,1] random varible}\\\\

GIBRAT
X\\sim\\mathrm{Gibrat}\\left(L,S\\right)\\\\
x\\in\\left(L,\\infty\\right)\\\\
L\\in\\mathbb{R},S\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=\\Phi\\left(\\ln x\\right)=\\frac{1}{2}\\left(1+\\mathrm{erf}\\left(\\frac{\\ln z(x)}{\\sqrt{2}}\\right)\\right)\\\\
f_{X}\\left(x\\right)=\\frac{1}{S}\\frac{1}{x\\sqrt{2\\pi}}\\exp\\left(-\\frac{1}{2}\\left(\\ln z(x)\\right)^{2}\\right)\\\\
F^{-1}_{X}\\left(u\\right)=L+S\\times \\exp\\left(\\Phi^{-1}\\left(u\\right)\\right)\\\\
\\mu'_{k}=E[X^k]=\\int_{L}^{\\infty}x^{k}f_{X}\\left(x\\right)dx=\\exp\\left(\\frac{k^{2}}{2}\\right)\\\\
\\mathrm{Mean}(X)=L+S\\mu'_{1}=L+S\\sqrt{e}\\\\
\\mathrm{Variance}(X)=S^{2}(\\mu'_{2}-\\mu'^{2}_{1})=S^{2}\\left[e{^2}-e\\right]\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}=\\sqrt{e-1}\\left(2+e\\right)\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}=e^{4}+2e^{3}+3e^{2}-3\\\\
\\mathrm{Median}(X)=L+S\\times \\exp\\left(\\Phi^{-1}\\left(1/2\\right)\\right)\\\\
\\mathrm{Mode}(X)=L+\\frac{S}{e}\\\\
L:\\text{Location parameter}\\\\
S:\\text{Scale parameter}\\\\
z\\left(x\\right)=\\left(x-L\\right)/S\\\\
u:\\text{Uniform[0,1] random varible}\\\\
\\Phi\\left(x\\right):\\text{Cumulative function from standard normal distribution}\\\\
\\Phi^{-1}\\left(x\\right):\\text{Inverse of cumulative function from standard normal distribution}\\\\
\\mathrm{erf}(x):\\text{Error function}\\\\

GUMBEL_LEFT
X\\sim\\mathrm{GumbelLeft}\\left(\\mu,\\sigma\\right)\\\\
x\\in\\left(-\\infty,\\infty\\right)\\\\
\\mu\\in\\mathbb{R},\\sigma\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=1-\\exp\\left(-e^{z(x)}\\right)\\\\
f_{X}\\left(x\\right)=\\frac{1}{\\sigma}\\exp\\left(z(x)-e^{z(x)}\\right)\\\\
F^{-1}_{X}\\left(u\\right)=\\mu+\\sigma\\ln\\left(-\\ln\\left(1-u\\right)\\right)\\\\
\\tilde{\\mu}'_{k}=E[\\tilde{X}^k]=\\int_{0}^{1}x^{k}f_{\\tilde{X}}\\left(x\\right)dx\\\\
\\mathrm{Mean}(X)=\\mu+\\sigma\\tilde{\\mu}'_{1}=\\mu-\\gamma\\sigma\\\\
\\mathrm{Variance}(X)=\\sigma^{2}(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})=\\sigma^{2}\\frac{\\pi^{2}}{6}\\\\
\\mathrm{Skewness}(X)=\\frac{\\tilde{\\mu}'_{3}-3\\tilde{\\mu}'_{2}\\tilde{\\mu}'_{1}+2\\tilde{\\mu}'^{3}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{1.5}}=-\\frac{12\\sqrt{6}\\zeta(3)}{\\pi^{3}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\tilde{\\mu}'_{4}-4\\tilde{\\mu}'_{1}\\tilde{\\mu}'_{3}+6\\tilde{\\mu}'^{2}_{1}\\tilde{\\mu}'_{2}-3\\tilde{\\mu}'^{4}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{2}}=3+\\frac{12}{5}\\\\
\\mathrm{Median}(X)=\\mu+\\sigma\\ln\\left(-\\ln\\left(\\frac{1}{2}\\right)\\right)\\\\
\\mathrm{Mode}(X)=\\mu\\\\
\\tilde{X}\\sim\\mathrm{GumbelLeft}\\left(0,1\\right)\\\\
\\mu:\\text{Location parameter}\\\\
\\sigma:\\text{Scale parameter}\\\\
z\\left(x\\right)=\\left(x-\\mu\\right)/\\sigma\\\\
u:\\text{Uniform[0,1] random varible}\\\\
\\gamma:\\text{Euler–Mascheroni constant}=0.5772156649\\\\
\\zeta(3):\\text{Apéry's constant}=1.2020569031\\\\

GUMBEL_RIGHT
X\\sim\\mathrm{GumbelRight}\\left(\\mu,\\sigma\\right)\\\\
x\\in\\left(-\\infty,\\infty\\right)\\\\
\\mu\\in\\mathbb{R},\\sigma\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=\\exp\\left(-e^{-z(x)}\\right)\\\\
f_{X}\\left(x\\right)=\\frac{1}{\\sigma}\\exp\\left(-\\left(z(x)+e^{-z(x)}\\right)\\right)\\\\
F^{-1}_{X}\\left(u\\right)=\\tilde{\\mu}-\\sigma\\ln\\left(-\\ln\\left(u\\right)\\right)\\\\
\\tilde{\\mu}'_{k}=E[\\tilde{X}^k]=\\int_{0}^{1}x^{k}f_{\\tilde{X}}\\left(x\\right)dx\\\\
\\mathrm{Mean}(X)=\\mu+\\sigma\\tilde{\\mu}'_{1}=\\mu+\\gamma\\sigma\\\\
\\mathrm{Variance}(X)=\\sigma^{2}(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})=\\sigma^{2}\\frac{\\pi^{2}}{6}\\\\
\\mathrm{Skewness}(X)=\\frac{\\tilde{\\mu}'_{3}-3\\tilde{\\mu}'_{2}\\tilde{\\mu}'_{1}+2\\tilde{\\mu}'^{3}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{1.5}}=\\frac{12\\sqrt{6}\\zeta(3)}{\\pi^{3}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\tilde{\\mu}'_{4}-4\\tilde{\\mu}'_{1}\\tilde{\\mu}'_{3}+6\\tilde{\\mu}'^{2}_{1}\\tilde{\\mu}'_{2}-3\\tilde{\\mu}'^{4}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{2}}=3+\\frac{12}{5}\\\\
\\mathrm{Median}(X)=\\mu-\\sigma\\ln\\left(-\\ln\\left(\\frac{1}{2}\\right)\\right)\\\\
\\mathrm{Mode}(X)=\\mu\\\\
\\tilde{X}\\sim\\mathrm{GumbelRight}\\left(0,1\\right)\\\\
\\mu:\\text{Location parameter}\\\\
\\sigma:\\text{Scale parameter}\\\\
z\\left(x\\right)=\\left(x-\\mu\\right)/\\sigma\\\\
u:\\text{Uniform[0,1] random varible}\\\\
\\gamma:\\text{Euler–Mascheroni constant}=0.5772156649\\\\
\\zeta(3):\\text{Apéry's constant}=1.2020569031\\\\

HALF_NORMAL
X\\sim\\mathrm{HalfNormal}\\left(\\mu,\\sigma\\right)\\\\
x\\in\\left(\\mu,\\infty\\right)\\\\
\\mu\\in\\mathbb{R},\\sigma\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=2\\Phi\\left(z(x)\\right)-1=\\operatorname{erf}\\left(\\frac{z(x)}{\\sqrt{2}}\\right)\\\\
f_{X}\\left(x\\right)=\\frac{\\sqrt{2}}{\\sigma\\sqrt{\\pi}}\\exp\\left(-\\frac{z(x)^2}{2}\\right)\\\\
F^{-1}_{X}\\left(u\\right)=\\mu+\\sigma\\Phi^{-1}\\left(\\frac{1+u}{2}\\right)=\\tilde{\\mu}+\\sigma\\sqrt{2}\\operatorname{erf}^{-1}(u)\\\\
\\tilde{\\mu}'_{k}=E[\\tilde{X}^k]=\\int_{0}^{1}x^{k}f_{\\tilde{X}}\\left(x\\right)dx=\\frac{2^{n/2} \\Gamma(\\frac{n+1}{2})}{\\sqrt{\\pi}}\\\\
\\mathrm{Mean}(X)=\\tilde{\\mu}+\\sigma\\tilde{\\mu}'_{1}=\\tilde{\\mu}+\\sigma\\sqrt{\\frac{2}{\\pi}}\\\\
\\mathrm{Variance}(X)=\\sigma^{2}(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})=\\sigma^2\\left(1-\\frac 2 \\pi\\right)\\\\
\\mathrm{Skewness}(X)=\\frac{\\tilde{\\mu}'_{3}-3\\tilde{\\mu}'_{2}\\tilde{\\mu}'_{1}+2\\tilde{\\mu}'^{3}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{1.5}}=\\frac{\\sqrt{2}(4-\\pi)}{(\\pi-2)^{3/2}}=0.9952717\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\tilde{\\mu}'_{4}-4\\tilde{\\mu}'_{1}\\tilde{\\mu}'_{3}+6\\tilde{\\mu}'^{2}_{1}\\tilde{\\mu}'_{2}-3\\tilde{\\mu}'^{4}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{2}}=3+\\frac{8(\\pi-3)}{(\\pi-2)^2}= 3.869177\\\\
\\mathrm{Median}(X)=\\mu+\\sigma\\sqrt{2}\\operatorname{erf}^{-1}(1/2)\\\\
\\mathrm{Mode}(X)=\\mu\\\\
\\tilde{X}\\sim\\mathrm{HalfNormal}\\left(0,1\\right)\\\\
\\mu:\\text{Location parameter}\\\\
\\sigma:\\text{Scale parameter}\\\\
z\\left(x\\right)=\\left(x-\\mu\\right)/\\sigma\\\\
u:\\text{Uniform[0,1] random varible}\\\\
\\Phi\\left(x\\right):\\text{Cumulative function from standard normal distribution}\\\\
\\Phi^{-1}\\left(x\\right):\\text{Inverse of cumulative function from standard normal distribution}\\\\
\\mathrm{erf}(x):\\text{Error function}\\\\
\\Gamma\\left(x\\right):\\text{Gamma function}\\\\

HYPERBOLIC_SECANT
X\\sim\\mathrm{HyperbolicSecant}\\left(\\mu,\\sigma\\right)\\\\
x\\in\\left(-\\infty,\\infty\\right)\\\\
\\mu\\in\\mathbb{R},\\sigma\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=\\frac{2}{\\pi} \\arctan\\left[\\exp\\!\\left(\\frac{\\pi}{2}\\,z(x)\\right)\\right]\\\\
f_{X}\\left(x\\right)=\\frac{1}{2\\sigma} \\operatorname{sech}\\!\\left(\\frac{\\pi}{2}\\,z(x)\\right)\\\\
F^{-1}_{X}\\left(u\\right)=\\mu+\\sigma\\frac{2}{\\pi}\\,\\ln\\!\\left[\\tan\\left(\\frac{\\pi}{2}\\,u\\right)\\right]\\\\
\\tilde{\\mu}'_{k}=E[\\tilde{X}^k]=\\int_{0}^{1}x^{k}f_{\\tilde{X}}\\left(x\\right)dx=\\frac{1+\\left(-1\\right)^{k}}{2\\pi2^{2k}}k!\\left[\\zeta\\left(k+1,\\frac{1}{4}\\right)-\\zeta\\left(k+1,\\frac{3}{4}\\right)\\right]\\\\
\\mathrm{Mean}(X)=\\mu+\\sigma\\tilde{\\mu}'_{1}=\\mu\\\\
\\mathrm{Variance}(X)=\\sigma^{2}(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})=\\sigma^{2}\\\\
\\mathrm{Skewness}(X)=\\frac{\\tilde{\\mu}'_{3}-3\\tilde{\\mu}'_{2}\\tilde{\\mu}'_{1}+2\\tilde{\\mu}'^{3}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{1.5}}=0\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\tilde{\\mu}'_{4}-4\\tilde{\\mu}'_{1}\\tilde{\\mu}'_{3}+6\\tilde{\\mu}'^{2}_{1}\\tilde{\\mu}'_{2}-3\\tilde{\\mu}'^{4}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{2}}=3\\\\
\\mathrm{Median}(X)=\\mu\\\\
\\mathrm{Mode}(X)=\\mu\\\\
\\tilde{X}\\sim\\mathrm{HyperbolicSecant}\\left(0,1\\right)\\\\
\\mu:\\text{Location parameter}\\\\
\\sigma:\\text{Scale parameter}\\\\
z\\left(x\\right)=\\left(x-\\mu\\right)/\\sigma\\\\
u:\\text{Uniform[0,1] random varible}\\\\
\\zeta(a,s):\\text{Hurwitz zeta function}\\\\

INVERSE_GAMMA
X\\sim\\mathrm{InverseGamma}\\left(\\alpha,\\beta\\right)\\\\
x\\in\\left(0,\\infty\\right)\\\\
\\alpha\\in\\mathbb{R}^{+},\\beta\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=1-\\frac{\\gamma(\\alpha,\\beta/x)}{\\Gamma(\\alpha)}=1-P\\left(\\alpha,\\frac{\\beta}{x}\\right)\\\\
f_{X}\\left(x\\right)=\\frac{\\beta^\\alpha}{\\Gamma(\\alpha)} x^{-\\alpha-1} \\exp\\left(-\\frac{\\beta}{x}\\right)\\\\
F^{-1}_{X}\\left(u\\right)=\\frac{\\beta}{P^{-1}\\left(\\alpha,1-u\\right)}\\\\
\\tilde{\\mu}'_{k}=E[\\tilde{X}^k]=\\int_{0}^{1}x^{k}f_{\\tilde{X}}\\left(x\\right)dx=\\frac{\\Gamma(\\alpha-k)}{\\Gamma(\\alpha)}=\\frac{1}{(\\alpha-1) \\cdots (\\alpha-k)}\\quad \\text{if } \\alpha>k\\\\
\\mathrm{Mean}(X)=\\beta\\tilde{\\mu}'_{1} \\\\
\\mathrm{Variance}(X)=\\beta^{2}(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})\\\\
\\mathrm{Skewness}(X)=\\frac{\\tilde{\\mu}'_{3}-3\\tilde{\\mu}'_{2}\\tilde{\\mu}'_{1}+2\\tilde{\\mu}'^{3}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{1.5}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\tilde{\\mu}'_{4}-4\\tilde{\\mu}'_{1}\\tilde{\\mu}'_{3}+6\\tilde{\\mu}'^{2}_{1}\\tilde{\\mu}'_{2}-3\\tilde{\\mu}'^{4}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{2}}\\\\
\\mathrm{Median}(X)=\\frac{\\beta}{P^{-1}\\left(\\alpha,\\frac{1}{2}\\right)}\\\\
\\mathrm{Mode}(X)=\\frac{\\beta}{\\alpha+1}\\\\
\\tilde{X}\\sim\\mathrm{InverseGamma}\\left(\\alpha,1\\right)\\\\
\\beta:\\text{Scale parameter}\\\\
u:\\text{Uniform[0,1] random varible}\\\\
P\\left(a,x\\right)=\\frac{\\gamma(a,x)}{\\Gamma(a)}:\\text{Regularized lower incomplete gamma function}\\\\
P^{-1}\\left(a,u\\right):\\text{Inverse of regularized lower incomplete gamma function}\\\\
\\gamma\\left(a,x\\right):\\text{Lower incomplete gamma function}\\\\
\\Gamma\\left(x\\right):\\text{Gamma function}\\\\

INVERSE_GAMMA_3P
X\\sim\\mathrm{InverseGamma_{3P}}\\left(\\alpha,\\beta,L\\right)\\\\
x\\in\\left(L,\\infty\\right)\\\\
\\alpha\\in\\mathbb{R}^{+},\\beta\\in\\mathbb{R}^{+},L\\in\\mathbb{R}\\\\
F_{X}\\left(x\\right)=1-\\frac{\\gamma(\\alpha,\\beta/(x-L))}{\\Gamma(\\alpha)}=1-P\\left(\\alpha,\\frac{\\beta}{x-L}\\right)\\\\
f_{X}\\left(x\\right)=\\frac{\\beta^\\alpha}{\\Gamma(\\alpha)} (x-L)^{-\\alpha-1} \\exp\\left(-\\frac{\\beta}{x-L}\\right)\\\\
F^{-1}_{X}\\left(u\\right)=L+\\frac{\\beta}{P^{-1}\\left(\\alpha,1-u\\right)}\\\\
\\tilde{\\mu}'_{k}=E[\\tilde{X}^k]=\\int_{0}^{1}x^{k}f_{\\tilde{X}}\\left(x\\right)dx=\\frac{\\Gamma(\\alpha-k)}{\\Gamma(\\alpha)}=\\frac{1}{(\\alpha-1) \\cdots (\\alpha-k)}\\quad \\text{if } \\alpha>k\\\\
\\mathrm{Mean}(X)=L+\\beta\\mu'_{1} \\\\
\\mathrm{Variance}(X)=\\beta^{2}(\\mu'_{2}-\\mu'^{2}_{1})\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}\\\\
\\mathrm{Median}(X)=L+\\frac{\\beta}{P^{-1}\\left(\\alpha,\\frac{1}{2}\\right)}\\\\
\\mathrm{Mode}(X)=L+\\frac{\\beta}{\\alpha+1}\\\\
\\tilde{X}\\sim\\mathrm{InverseGamma_{3P}}\\left(\\alpha,1,0\\right)\\\\
L:\\text{Location parameter}\\\\
\\beta:\\text{Scale parameter}\\\\
u:\\text{Uniform[0,1] random varible}\\\\
P\\left(a,x\\right)=\\frac{\\gamma(a,x)}{\\Gamma(a)}:\\text{Regularized lower incomplete gamma function}\\\\
P^{-1}\\left(a,u\\right):\\text{Inverse of regularized lower incomplete gamma function}\\\\
\\gamma\\left(a,x\\right):\\text{Lower incomplete gamma function}\\\\
\\Gamma\\left(x\\right):\\text{Gamma function}\\\\

INVERSE_GAUSSIAN
X\\sim\\mathrm{InverseGaussian}\\left(\\mu,\\lambda\\right)\\\\
x\\in\\left(0,\\infty\\right)\\\\
\\mu\\in\\mathbb{R}^{+},\\lambda\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=\\Phi\\left(\\sqrt{\\frac{\\lambda}{x}}\\left(\\frac{x}{\\mu}-1\\right)\\right)+\\exp\\left(\\frac{2 \\lambda}{\\mu}\\right) \\Phi\\left(-\\sqrt{\\frac{\\lambda}{x}}\\left(\\frac{x}{\\mu}+1\\right)\\right)\\\\
f_{X}\\left(x\\right)=\\sqrt\\frac{\\lambda}{2 \\pi x^3} \\exp\\left[-\\frac{\\lambda (x-\\mu)^2}{2 \\mu^2 x}\\right]\\\\
Sample_{X}=\\left\\{\\begin{array}{cl} x_{0} \\quad \\text{if } u_{2}\\leqslant\\frac{\\mu}{\\mu+x_{0}}\\\\ \\frac{\\mu^{2}}{x_{0}} \\quad \\text{if } u_{2}\\geqslant \\frac{\\mu}{\\mu+x_{0}} \\end{array} \\right.\\\\
\\mu'_{k}=E[X^k]=\\int_{0}^{\\infty }x^{k}f_{X}\\left(x\\right)dx\\\\
\\mathrm{Mean}(X)=\\mu'_{1}=\\mu\\\\
\\mathrm{Variance}(X)=\\mu'_{2}-\\mu'^{2}_{1}=\\frac{\\mu^3}{\\lambda}\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}=3\\left(\\frac{\\mu}{\\lambda}\\right)^{1/2}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}=3+\\frac{15 \\mu}{\\lambda}\\\\
\\mathrm{Median}(X)=F^{-1}_{X}\\left(\\frac{1}{2}\\right)\\\\
\\mathrm{Mode}(X)=\\mu\\left[\\left(1+\\frac{9 \\mu^2}{4 \\lambda^2}\\right)^\\frac{1}{2}-\\frac{3 \\mu}{2 \\lambda}\\right]\\\\
\\text{It is not possible to compute an analytic function for the inverse of the cumulative}\\\\
\\text{density function. However,we can compute a random sample from the distribution.}\\\\
\\Phi\\left(x\\right):\\text{Cumulative function from standard normal distribution}\\\\
\\Phi^{-1}\\left(x\\right):\\text{Inverse of cumulative function from standard normal distribution}\\\\
x_{0}=\\mu+\\frac{\\mu^2 [\\Phi^{-1}\\left(u_{1}\\right)]^{2}}{2\\lambda}-\\frac{\\mu}{2\\lambda}\\sqrt{4\\mu \\lambda [\\Phi^{-1}\\left(u_{1}\\right)]^{2}+\\mu^2 ([\\Phi^{-1}\\left(u_{1}\\right)]^{2})^2}\\\\
u_{1}:\\text{Uniform[0,1] random varible}\\\\
u_{2}:\\text{Uniform[0,1] random varible}\\\\

INVERSE_GAUSSIAN_3P
X\\sim\\mathrm{InverseGaussian_{3P}}\\left(\\mu,\\lambda,L\\right)\\\\
x\\in\\left(0,\\infty\\right)\\\\
\\mu\\in\\mathbb{R}^{+},\\lambda\\in\\mathbb{R}^{+},L\\in\\mathbb{R}\\\\
F_{X}\\left(x\\right)=\\Phi\\left(\\sqrt{\\frac{\\lambda}{x-L}}\\left(\\frac{x-L}{\\mu}-1\\right)\\right)+\\exp\\left(\\frac{2 \\lambda}{\\mu}\\right) \\Phi\\left(-\\sqrt{\\frac{\\lambda}{x-L}}\\left(\\frac{x-L}{\\mu}+1\\right)\\right)\\\\
f_{X}\\left(x\\right)=\\sqrt\\frac{\\lambda}{2 \\pi (x-L)^3} \\exp\\left[-\\frac{\\lambda (x-\\mu-L)^2}{2 \\mu^2 (x-L)}\\right]\\\\
Sample_{X}=\\left\\{\\begin{array}{cl} L+x_{0} \\quad \\text{if} \\quad u_{2}\\leqslant\\frac{\\mu}{\\mu+x_{0}}\\\\ L+\\frac{\\mu^{2}}{x_{0}} \\quad \\text{if} \\quad u_{2}\\geqslant \\frac{\\mu}{\\mu+x_{0}} \\end{array} \\right.\\\\
\\mu'_{k}=E[X^k]=\\int_{L}^{\\infty }x^{k}f_{X}\\left(x\\right)dx\\\\
\\mathrm{Mean}(X)=\\mu'_{1}=L+\\mu\\\\
\\mathrm{Variance}(X)=\\mu'_{2}-\\mu'^{2}_{1}=\\frac{\\mu^3}{\\lambda}\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}=3\\left(\\frac{\\mu}{\\lambda}\\right)^{1/2}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}=3+\\frac{15 \\mu}{\\lambda}\\\\
\\mathrm{Median}(X)=F^{-1}_{X}\\left(\\frac{1}{2}\\right)\\\\
\\mathrm{Mode}(X)=L+\\mu\\left[\\left(1+\\frac{9 \\mu^2}{4 \\lambda^2}\\right)^\\frac{1}{2}-\\frac{3 \\mu}{2 \\lambda}\\right]\\\\
\\text{It is not possible to compute an analytic function for the inverse of the cumulative}\\\\
\\text{density function. However,we can compute a random sample from the distribution.}\\\\
L:\\text{Location parameter}\\\\
\\Phi\\left(x\\right):\\text{Cumulative function from standard normal distribution}\\\\
\\Phi^{-1}\\left(x\\right):\\text{Inverse of cumulative function from standard normal distribution}\\\\
x_{0}=\\mu+\\frac{\\mu^2 [\\Phi^{-1}\\left(u_{1}\\right)]^{2}}{2\\lambda}-\\frac{\\mu}{2\\lambda}\\sqrt{4\\mu \\lambda [\\Phi^{-1}\\left(u_{1}\\right)]^{2}+\\mu^2 ([\\Phi^{-1}\\left(u_{1}\\right)]^{2})^2}\\\\
u_{1}:\\text{Uniform[0,1] random varible}\\\\
u_{2}:\\text{Uniform[0,1] random varible}\\\\

JOHNSON_SB
X\\sim\\mathrm{JohnsonSB}\\left(\\delta,\\lambda,\\gamma,\\xi \\right )\\\\
x\\in\\left(\\xi,\\xi+\\lambda\\right)\\\\
\\delta\\in\\mathbb{R}^{+},\\lambda\\in\\mathbb{R}^{+},\\gamma\\in\\mathbb{R},\\xi\\in\\mathbb{R}\\\\
F_{X}\\left(x\\right)=\\Phi\\left(\\gamma+\\delta\\ln\\frac{z(x)}{1-z(x)}\\right)\\\\
f_{X}\\left(x\\right)=\\frac{\\delta}{\\lambda\\sqrt{2\\pi}z(1-z(x))}\\exp\\left[-\\frac{1}{2}\\left(\\gamma+\\delta\\ln\\frac{z(x)}{1-z(x)}\\right)^2\\right]\\\\
F^{-1}_{X}\\left(u\\right)=\\frac{\\lambda\\exp\\left(\\frac{\\Phi^{-1}(u)-\\gamma}{\\delta}\\right)}{1+\\exp\\left(\\frac{\\Phi^{-1}(u)-\\gamma}{\\delta}\\right)}+\\xi\\\\
\\mu'_{k}=E[X^k]=\\int_{\\xi}^{\\xi+\\lambda }x^{k}f_{X}\\left(x\\right)dx\\\\
\\mathrm{Mean}(X)=\\mu'_{1}\\\\
\\mathrm{Variance}(X)=\\mu'_{2}-\\mu'^{2}_{1}\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}\\\\
\\mathrm{Median}(X)=\\frac{\\lambda\\exp\\left(\\frac{\\Phi^{-1}\\left(1/2\\right)-\\gamma}{\\delta}\\right)}{1+\\exp\\left(\\frac{\\Phi^{-1}\\left(1/2\\right)-\\gamma}{\\delta}\\right)}+\\xi\\\\
\\mathrm{Mode}(X)=\\arg\\max_{x}f_{X}\\left(x\\right)\\\\
\\xi:\\text{Location parameter}\\\\
\\lambda:\\text{Scale parameter}\\\\
z\\left(x\\right)=\\left(x-\\xi\\right)/\\lambda\\\\
u:\\text{Uniform[0,1] random varible}\\\\
\\Phi\\left(x\\right):\\text{Cumulative function from standard normal distribution}\\\\
\\Phi^{-1}\\left(x\\right):\\text{Inverse of cumulative function from standard normal distribution}\\\\

JOHNSON_SU
X\\sim\\mathrm{JohnsonSU}\\left(\\delta,\\lambda,\\gamma,\\xi\\right)\\\\
x\\in\\left(-\\infty,\\infty\\right)\\\\
\\delta\\in\\mathbb{R}^{+},\\lambda\\in\\mathbb{R}^{+},\\gamma\\in\\mathbb{R},\\xi\\in\\mathbb{R}\\\\
F_{X}\\left(x\\right)=\\Phi\\left(\\gamma+\\delta\\sinh^{-1}(z(x))\\right)\\\\
f_{X}\\left(x\\right)=\\frac{\\delta}{\\lambda\\sqrt{2\\pi}\\sqrt{z(x)^2+1}}\\exp\\left[-\\frac{1}{2}\\left(\\gamma+\\delta\\sinh^{-1}(z(x))\\right)^2\\right]\\\\
F^{-1}_{X}\\left(u\\right)=\\lambda\\sinh\\left(\\frac{\\Phi^{-1}(u)-\\gamma}{\\delta}\\right)+\\xi\\\\
\\mu'_{k}=E[X^k]=\\int_{-\\infty }^{\\infty }x^{k}f_{X}\\left(x\\right)dx\\\\
\\mathrm{Mean}(X)=\\mu'_{1}=\\xi-\\lambda \\exp\\frac{\\delta^{-2}}{2} \\sinh\\left(\\frac{\\gamma}{\\delta}\\right)\\\\
\\mathrm{Variance}(X)=\\mu'_{2}-\\mu'^{2}_{1}=\\frac{\\lambda^2}{2} (\\exp(\\delta^{-2})-1)\\left(\\exp(\\delta^{-2}) \\cosh\\left(\\frac{2\\gamma}{\\delta}\\right) +1\\right)\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}=-\\frac{\\lambda^{3}\\sqrt{e^{\\delta^{-2}}}(e^{\\delta^{-2}}-1)^{2}(e^{\\delta^{-2}})(e^{\\delta^{-2}}+2)\\sinh(\\frac{3\\gamma}{\\delta})+3\\sinh(\\frac{2\\gamma}{\\delta}))}{4\\mathrm{Variance}(X)^{1.5}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}=\\frac{\\lambda^{4}(e^{\\delta^{-2}}-1)^{2}(K_{1}+K_{2}+K_{3})}{8\\mathrm{Variance}(X)^{2}}\\\\
\\mathrm{Median}(X)=\\xi+\\lambda \\sinh\\left(-\\frac{\\gamma}{\\delta}\\right)\\\\
\\mathrm{Mode}(X)=\\arg\\max_{x}f_{X}\\left(x\\right)\\\\
\\xi:\\text{Location parameter}\\\\
\\lambda:\\text{Scale parameter}\\\\
z\\left(x\\right)=\\left(x-\\xi\\right)/\\lambda\\\\
u:\\text{Uniform[0,1] random varible}\\\\
\\Phi\\left(x\\right):\\text{Cumulative function from standard normal distribution}\\\\
\\Phi^{-1}\\left(x\\right):\\text{Inverse of cumulative function from standard normal distribution}\\\\
K_{1}=\\left(e^{\\delta^{-2}}\\right)^{2}\\left(\\left(e^{\\delta^{-2}}\\right)^{4}+2\\left(e^{\\delta^{-2}}\\right)^{3}+3\\left(e^{\\delta^{-2}}\\right)^{2}-3\\right)\\cosh\\left(\\frac{4\\gamma}{\\delta}\\right)\\\\
K_{2}=4\\left(e^{\\delta^{-2}}\\right)^{2}\\left(\\left(e^{\\delta^{-2}}\\right)+2\\right)\\cosh\\left(\\frac{3\\gamma}{\\delta}\\right)\\\\
K_{3}=3\\left(2\\left(e^{\\delta^{-2}}\\right)+1\\right)\\\\

KUMARASWAMY
X\\sim\\mathrm{Kumaraswamy}\\left(\\alpha,\\beta,min,max\\right)\\\\
x\\in\\left(min,max\\right)\\\\
\\alpha\\in\\mathbb{R}^{+},\\beta\\in\\mathbb{R}^{+},min\\in\\mathbb{R},max\\in\\mathbb{R}\\\\
F_{X}\\left(x\\right)=1-(1-z(x)^\\alpha)^\\beta\\\\
f_{X}\\left(x\\right)=\\alpha \\beta z(x)^{\\alpha-1}(1-z(x)^\\alpha)^{\\beta-1}\\\\
F^{-1}_{X}\\left(u\\right)=min+\\left(max-min\\right)\\times (1-(1-u)^\\frac{1}{\\beta})^\\frac{1}{\\alpha}\\\\
\\tilde{\\mu}'_{k}=E[\\tilde{X}^k]=\\int_{0}^{1}x^{k}f_{\\tilde{X}}\\left(x\\right)dx=\\beta Beta(1+\\frac{k}{\\alpha},\\beta)\\\\
\\mathrm{Mean}(X)=min+\\left(max-min\\right)\\times \\tilde{\\mu}'_{1}\\\\
\\mathrm{Variance}(X)=\\left(max-min\\right)^{2}(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})\\\\
\\mathrm{Skewness}(X)=\\frac{\\tilde{\\mu}'_{3}-3\\tilde{\\mu}'_{2}\\tilde{\\mu}'_{1}+2\\tilde{\\mu}'^{3}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{1.5}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\tilde{\\mu}'_{4}-4\\tilde{\\mu}'_{1}\\tilde{\\mu}'_{3}+6\\tilde{\\mu}'^{2}_{1}\\tilde{\\mu}'_{2}-3\\tilde{\\mu}'^{4}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{2}}\\\\
\\mathrm{Median}(X)=\\left(1-2^{-1/b}\\right)^{1/a}\\\\
\\mathrm{Mode}(X)=\\left(\\frac{a-1}{ab-1}\\right)^{1/a}\\\\
\\tilde{X}\\sim\\mathrm{Kumaraswamy}\\left(\\alpha,\\beta,0,1\\right)\\\\
z\\left(x\\right)=\\left(x-min\\right)/\\left(max-min\\right)\\\\
u:\\text{Uniform[0,1] random varible}\\\\
Beta\\left(x,y\\right):\\text{Beta function}\\\\

LAPLACE
X\\sim\\mathrm{Laplace}\\left(\\mu,b\\right)\\\\
x\\in\\left(-\\infty,\\infty\\right)\\\\
\\mu\\in\\mathbb{R}^{+},b\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=\\tfrac{1}{2}+\\tfrac{1}{2} \\mathrm{sign}(x-\\mu)\\left(1-\\exp\\left(-\\frac{|x-\\mu|}{b} \\right ) \\right )\\\\
f_{X}\\left(x\\right)=\\frac{1}{2b} \\exp\\left(-\\frac{|x-\\mu|}{b}\\right)\\\\
F^{-1}_{X}\\left(u\\right)=\\mu-b\\times \\mathrm{sign}\\left(p-\\frac{1}{2}\\right)\\,\\ln\\left(1-2\\left|p-\\frac{1}{2}\\right|\\right)\\\\
\\mu'_{k}=E[X^k]=\\int_{-\\infty}^{\\infty}x^{k}f_{X}\\left(x\\right)dx=\\bigg({\\frac{1}{2}}\\bigg) \\sum_{k=0}^r \\bigg[{\\frac{r!}{(r-k)!}} b^k \\mu^{(r-k)} \\{1+(-1)^k\\}\\bigg]\\\\
\\mathrm{Mean}(X)=\\mu'_{1}=\\mu\\\\
\\mathrm{Variance}(X)=\\mu'_{2}-\\mu'^{2}_{1}=2b^2\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}=0\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}=6\\\\
\\mathrm{Median}(X)=\\mu\\\\
\\mathrm{Mode}(X)=\\mu\\\\
\\mu:\\text{Location parameter}\\\\
b:\\text{Scale parameter}\\\\
u:\\text{Uniform[0,1] random varible}\\\\

LEVY
X\\sim\\mathrm{Levy}\\left(\\mu,c\\right)\\\\
x\\in [\\mu,\\infty)\\\\
\\mu\\in\\mathbb{R},c\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=1-\\textrm{erf}\\left(\\sqrt{\\frac{c}{2(x-\\mu)}}\\right)\\\\
f_{X}\\left(x\\right)=\\sqrt{\\frac{c}{2\\pi}}~~\\frac{e^{-\\frac{c}{2(x-\\mu)}}}{(x-\\mu)^{3/2}}\\\\
F^{-1}_{X}\\left(u\\right)=\\mu+\\frac{c}{2\\left(\\textrm{erf}^{-1}(1-u)\\right)^2}\\\\
\\mu'_{k}=E[X^k]=\\int_{\\mu }^{\\infty }x^{k}f_{X}\\left(x\\right)dx\\\\
\\mathrm{Mean}(X)=\\mu'_{1}=\\infty\\\\
\\mathrm{Variance}(X)=\\mu'_{2}-\\mu'^{2}_{1}=\\infty\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}=\\text{undefined}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}=\\text{undefined}\\\\
\\mathrm{Median}(X)=\\mu+\\frac{c}{2(\\textrm{erf}^{-1}(1/2))^2}\\\\
\\mathrm{Mode}(X)=\\mu+\\frac{c}{3}\\\\
\\mu:\\text{Location parameter}\\\\
c:\\text{Scale parameter}\\\\
u:\\text{Uniform[0,1] random varible}\\\\
\\mathrm{erf}(x):\\text{Error function}\\\\
\\mathrm{erf}^{-1}(x):\\text{Inverse of error function}\\\\

LOGGAMMA
X\\sim\\mathrm{LogGamma}\\left(c,\\mu,\\sigma\\right)\\\\
x\\in\\left(0,\\infty\\right)\\\\
c\\in\\mathbb{R}^{+},\\mu\\in\\mathbb{R},\\sigma\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=\\frac{\\gamma\\left(c,e^{x}\\right)}{\\Gamma\\left(c\\right)}=P\\left(c,e^{z(x)}\\right)\\\\
f_{X}\\left(x\\right)=\\frac{\\exp\\left(cz(x)-e^{z(x)}\\right)}{\\sigma\\Gamma\\left(c\\right)}\\\\
F^{-1}_{X}\\left(u\\right)=\\mu+\\sigma\\ln\\left(P^{-1}\\left(u,c\\right)\\right)\\\\
\\mu'_{k}=E[X^k]=\\int_{-\\infty }^{\\infty }x^{k}f_{X}\\left(x\\right)dx\\\\
\\mathrm{Mean}(X)=\\mu'_{1}=\\mu+\\sigma\\psi_{0}\\\\
\\mathrm{Variance}(X)=\\mu'_{2}-\\mu'^{2}_{1}=\\alpha^{2}\\psi_{1}\\left(c\\right)\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}=\\frac{\\psi_{2}\\left(c\\right)}{\\psi_{1}\\left(c\\right)}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}=\\frac{\\psi_{3}\\left(c\\right)}{\\psi_{1}\\left(c\\right)}\\\\
\\mathrm{Median}(X)=\\mu+\\sigma\\ln\\left(P^{-1}\\left(1/2,c\\right)\\right)\\\\
\\mathrm{Mode}(X)=\\mu+\\sigma\\ln(c)\\\\
\\mu:\\text{Location parameter}\\\\
\\sigma:\\text{Scale parameter}\\\\
z\\left(x\\right)=\\left(x-\\mu\\right)/\\sigma\\\\
u:\\text{Uniform[0,1] random varible}\\\\
P\\left(a,x\\right)=\\frac{\\gamma(a,x)}{\\Gamma(a)}:\\text{Regularized lower incomplete gamma function}\\\\
P^{-1}\\left(a,u\\right):\\text{Inverse of regularized lower incomplete gamma function}\\\\
\\gamma\\left(a,x\\right):\\text{Lower incomplete gamma function}\\\\
\\Gamma\\left(x\\right):\\text{Gamma function}\\\\
\\psi_{0}\\left(x\\right):\\text{Digamma function}\\\\
\\psi_{n}\\left(x\\right):\\text{Polygamma function of order }n\\in\\mathbb{N}\\\\

LOGISTIC
X\\sim\\mathrm{Logistic}\\left(\\mu,\\sigma\\right)\\\\
x\\in\\left(-\\infty,\\infty\\right)\\\\
\\mu\\in\\mathbb{R},\\sigma\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=\\frac{1}{1+e^{-(x-\\mu)/\\sigma}}\\\\
f_{X}\\left(x\\right)=\\frac{e^{-(x-\\mu)/\\sigma}} {\\sigma\\left(1+e^{-(x-\\mu)/\\sigma}\\right)^2}\\\\
F^{-1}_{X}\\left(u\\right)=\\mu+\\sigma \\log\\left(\\frac{u}{1-u}\\right)\\\\
\\mu'_{k}=E[X^k]=\\int_{-\\infty }^{\\infty }x^{k}f_{X}\\left(x\\right)dx\\\\
\\mathrm{Mean}(X)=\\mu'_{1}=\\mu\\\\
\\mathrm{Variance}(X)=\\mu'_{2}-\\mu'^{2}_{1}=\\frac{\\sigma^2 \\pi^2}{3}\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}=0\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}=3+6/5\\\\
\\mathrm{Median}(X)=\\mu\\\\
\\mathrm{Mode}(X)=\\mu\\\\
\\mu:\\text{Location parameter}\\\\
\\sigma:\\text{Scale parameter}\\\\
u:\\text{Uniform[0,1] random varible}\\\\

LOGLOGISTIC
X\\sim\\mathrm{LogLogistic}\\left(\\alpha,\\beta\\right)\\\\
x\\in [0,\\infty)\\\\
\\alpha\\in\\mathbb{R}^{+},\\beta\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)={ 1 \\over 1+(x/\\alpha)^{-\\beta} }\\\\
f_{X}\\left(x\\right)=\\frac{ (\\beta/\\alpha)(x/\\alpha)^{\\beta-1} }{ \\left (1+(x/\\alpha)^{\\beta}\\right)^2  }\\\\
F^{-1}_{X}\\left(u\\right)=\\alpha\\left(\\frac{u}{1-u}\\right)^{1/\\beta}\\\\
\\mu'_{k}=E[X^k]=\\int_{0}^{\\infty}x^{k}f_{X}\\left(x\\right)dx=\\alpha^k Beta(1-k/\\beta,1+k/\\beta)=\\alpha^k\\,{k\\pi/\\beta \\over \\sin(k\\pi/\\beta)}\\\\
\\mathrm{Mean}(X)=\\mu'_{1}\\\\
\\mathrm{Variance}(X)=\\mu'_{2}-\\mu'^{2}_{1}\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}\\\\
\\mathrm{Median}(X)=\\alpha\\\\
\\mathrm{Mode}(X)=\\alpha\\left(\\frac{\\beta-1}{\\beta+1}\\right)^{1/\\beta}\\\\
\\alpha:\\text{Scale parameter}\\\\
u:\\text{Uniform[0,1] random varible}\\\\
Beta\\left(x,y\\right):\\text{Beta function}\\\\

LOGLOGISTIC_3P
X\\sim\\mathrm{LogLogistic_{3P}}\\left(\\alpha,\\beta,L\\right)\\\\
x\\in [L,\\infty)\\\\
\\alpha\\in\\mathbb{R}^{+},\\beta\\in\\mathbb{R}^{+},L\\in\\mathbb{R}\\\\
F_{X}\\left(x\\right)={ 1 \\over 1+((x-L)/\\alpha)^{-\\beta} }\\\\
f_{X}\\left(x\\right)=\\frac{ (\\beta/\\alpha)((x-L)/\\alpha)^{\\beta-1} }{ \\left (1+((x-L)/\\alpha)^{\\beta}\\right)^2  }\\\\
F^{-1}_{X}\\left(u\\right)=L+\\alpha\\left(\\frac{u}{1-u}\\right)^{1/\\beta}\\\\
\\tilde{\\mu}'_{k}=E[\\tilde{X}^k]=\\int_{0}^{\\infty}x^{k}f_{\\tilde{X}}\\left(x\\right)dx=\\alpha^k Beta(1-k/\\beta,1+k/\\beta)=\\alpha^k\\,{k\\pi/\\beta \\over \\sin(k\\pi/\\beta)}\\\\
\\mathrm{Mean}(X)=L+\\tilde{\\mu}'_{1}\\\\
\\mathrm{Variance}(X)=\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1}\\\\
\\mathrm{Skewness}(X)=\\frac{\\tilde{\\mu}'_{3}-3\\tilde{\\mu}'_{2}\\tilde{\\mu}'_{1}+2\\tilde{\\mu}'^{3}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{1.5}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\tilde{\\mu}'_{4}-4\\tilde{\\mu}'_{1}\\tilde{\\mu}'_{3}+6\\tilde{\\mu}'^{2}_{1}\\tilde{\\mu}'_{2}-3\\tilde{\\mu}'^{4}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{2}}\\\\
\\mathrm{Median}(X)=L+\\alpha\\\\
\\mathrm{Mode}(X)=L+\\alpha\\left(\\frac{\\beta-1}{\\beta+1}\\right)^{1/\\beta}\\\\
\\tilde{X}\\sim\\mathrm{LogLogistic}\\left(\\alpha,\\beta\\right)\\\\
L:\\text{Location parameter}\\\\
\\alpha:\\text{Scale parameter}\\\\
u:\\text{Uniform[0,1] random varible}\\\\
Beta\\left(x,y\\right):\\text{Beta function}\\\\

LOGNORMAL
X\\sim\\mathrm{LogNormal}\\left(\\mu,\\sigma\\right)\\\\
x\\in\\left(-\\infty,\\infty\\right)\\\\
\\mu\\in\\mathbb{R},\\sigma\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=\\frac{1}{2}\\left[1+\\operatorname{erf}\\left(\\frac{\\ln (x)-\\mu}{\\sigma\\sqrt{2}}\\right)\\right]\\\\
f_{X}\\left(x\\right)=\\frac 1 {x\\sigma\\sqrt{2\\pi}}\\ \\exp\\left(-\\frac{\\left(\\ln\\left(x\\right) -\\mu\\right)^2}{2\\sigma^2}\\right)\\\\
F^{-1}_{X}\\left(u\\right)=\\exp(\\mu+\\sqrt{2\\sigma^2}\\operatorname{erf}^{-1}(2u-1))\\\\
\\mu'_{k}=E[X^k]=\\int_{-\\infty}^{\\infty}x^{k}f_{X}\\left(x\\right)dx=e^{k\\mu+k^2\\sigma^2/2}\\\\
\\mathrm{Mean}(X)=\\mu'_{1}\\\\
\\mathrm{Variance}(X)=\\mu'_{2}-\\mu'^{2}_{1}\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}\\\\
\\mathrm{Median}(X)=\\exp(\\mu)\\\\
\\mathrm{Mode}(X)=\\exp(\\mu-\\sigma^2)\\\\
\\mu:\\text{Location parameter}\\\\
\\sigma:\\text{Scale parameter}\\\\
u:\\text{Uniform[0,1] random varible}\\\\

MAXWELL
X\\sim\\mathrm{Maxwell}\\left(\\alpha,L\\right)\\\\
x\\in\\left(0,\\infty\\right)\\\\
\\alpha\\in\\mathbb{R}^{+},L\\in\\mathbb{R}\\\\
F_{X}\\left(x\\right)=\\operatorname{erf}\\left(\\frac{x-L}{\\sqrt{2} \\alpha}\\right) -\\sqrt{\\frac{2}{\\pi}}\\frac{(x-L) e^{-(x-L)^2/\\left(2\\alpha^2\\right)}}{\\alpha}\\\\
f_{X}\\left(x\\right)=\\sqrt{\\frac{2}{\\pi}}\\frac{(x-L)^2 e^{-(x-L)^2/\\left(2\\alpha^2\\right)}}{\\alpha^3}\\\\
F^{-1}_{X}\\left(u\\right)=L+\\alpha\\sqrt{2P^{-1}\\left(1.5,u\\right)}\\\\
\\mu'_{k}=E[X^k]=\\int_{-\\infty}^{\\infty}x^{k}f_{X}\\left(x\\right)dx\\\\
\\mathrm{Mean}(X)=\\mu'_{1}=L+2\\alpha \\sqrt{\\frac{2}{\\pi}}\\\\
\\mathrm{Variance}(X)=\\mu'_{2}-\\mu'^{2}_{1}=\\frac{\\alpha^2(3 \\pi-8)}{\\pi}\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}=\\frac{2 \\sqrt{2} (16 -5 \\pi)}{(3 \\pi-8)^{3/2}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}=4\\frac{\\left(-96+40\\pi-3\\pi^2\\right)}{(3 \\pi-8)^2}+3\\\\
\\mathrm{Median}(X)=L+\\alpha\\sqrt{2P^{-1}\\left(1.5,\\frac{1}{2}\\right)}\\\\
\\mathrm{Mode}(X)=L+\\alpha\\sqrt{2}\\\\
L:\\text{Location parameter}\\\\
\\alpha:\\text{Scale parameter}\\\\
u:\\text{Uniform[0,1] random varible}\\\\
P^{-1}\\left(a,u\\right):\\text{Inverse of regularized lower incomplete gamma function}\\\\

MOYAL
X\\sim\\mathrm{Moyal}\\left(\\mu,\\sigma\\right)\\\\
x\\in\\left(-\\infty,\\infty\\right)\\\\
\\mu\\in\\mathbb{R},\\sigma\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=1-P\\left(\\frac{1}{2},\\frac{e^{-z(x)}}{2}\\right)=1-\\mathrm{erf}\\left(\\frac{\\exp\\left(-0.5z(x)\\right)}{\\sqrt{2}}\\right)\\\\
f_{X}\\left(x\\right)=\\frac{1}{\\sqrt{2\\pi}}\\exp\\left(-\\frac{1}{2}\\left(z(x)+e^{-z(x)}\\right)\\right)\\\\
F^{-1}_{X}\\left(u\\right)=\\mu+\\sigma\\ln\\left[\\Phi^{-1}\\left(\\left(\\frac{1-u}{2}\\right)^{2}\\right)\\right]=\\mu+\\sigma\\ln\\left[2P^{-1}\\left(\\frac{1}{2},1-u\\right)\\right]\\\\
\\mu'_{k}=E[X^k]=\\int_{-\\infty }^{\\infty }x^{k}f_{X}\\left(x\\right)dx\\\\
\\mathrm{Mean}(X)=\\mu'_{1}=\\mu+\\sigma(\\ln(2)+\\gamma)\\\\
\\mathrm{Variance}(X)=\\mu'_{2}-\\mu'^{2}_{1}=\\sigma^{2}\\left(\\frac{\\pi^{2}}{2}\\right)\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}=\\frac{28\\sqrt{2}\\zeta(3)}{\\pi^{3}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}=7\\\\
\\mathrm{Median}(X)=\\mu+\\sigma\\ln\\left[2P^{-1}\\left(\\frac{1}{2},\\frac{1}{2}\\right)\\right]\\\\
\\mathrm{Mode}(X)=\\mu\\\\
\\mu:\\text{Location parameter}\\\\
\\sigma:\\text{Scale parameter}\\\\
z\\left(x\\right)=\\left(x-\\mu\\right)/\\sigma\\\\
P\\left(a,x\\right)=\\frac{\\gamma(a,x)}{\\Gamma(a)}:\\text{Regularized lower incomplete gamma function}\\\\
P^{-1}\\left(a,u\\right):\\text{Inverse of regularized lower incomplete gamma function}\\\\
\\gamma\\left(a,x\\right):\\text{Lower incomplete gamma function}\\\\
\\Gamma\\left(x\\right):\\text{Gamma function}\\\\
\\mathrm{erf}(x):\\text{Error function}\\\\
\\Phi^{-1}\\left(x\\right):\\text{Inverse of cumulative function from standard normal distribution}\\\\
P\\left(a,x\\right)=\\frac{\\gamma(a,x)}{\\Gamma(a)}:\\text{Regularized lower incomplete gamma function}\\\\
P^{-1}\\left(a,u\\right):\\text{Inverse of regularized lower incomplete gamma function}\\\\
\\gamma\\left(a,x\\right):\\text{Lower incomplete gamma function}\\\\
\\Gamma\\left(x\\right):\\text{Gamma function}\\\\
\\gamma:\\text{Euler–Mascheroni constant}=0.5772156649\\\\
\\zeta(3):\\text{Apéry's constant}=1.2020569031\\\\

NAKAGAMI
X\\sim\\mathrm{Nakagami}\\left(m,\\Omega\\right)\\\\
x\\in\\left(0,\\infty\\right)\\\\
m\\in\\mathbb{R}^{+}_{\\geqslant frac{1}{2}},\\Omega\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=\\frac{\\gamma\\left(m,\\frac{m}{\\Omega} x^2\\right)}{\\Gamma(m)}=P\\left(m,\\frac{m}{\\Omega} x^2\\right)\\\\
f_{X}\\left(x\\right)=\\frac{2m^m}{\\Gamma(m)\\Omega^m} x^{2m-1} \\exp\\left(-\\frac{m}{\\Omega}x^2\\right)\\\\
F^{-1}_{X}\\left(u\\right)=\\sqrt{\\frac{\\Omega}{m}P^{-1}\\left(m,u\\right)}\\\\
\\mu'_{k}=E[X^k]=\\int_{-\\infty }^{\\infty }x^{k}f_{X}\\left(x\\right)dx\\\\
\\mathrm{Mean}(X)=\\mu'_{1}=\\frac{\\Gamma(m+\\frac{1}{2})}{\\Gamma(m)}\\left(\\frac{\\Omega}{m}\\right)^{1/2}\\\\
\\mathrm{Variance}(X)=\\mu'_{2}-\\mu'^{2}_{1}=\\Omega\\left(1-\\frac{1}{m}\\left(\\frac{\\Gamma(m+\\frac{1}{2})}{\\Gamma(m)}\\right)^2\\right)\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}=\\frac{\\frac{\\Gamma(m+\\frac{1}{2})}{\\Gamma(m)\\sqrt{m}}\\left(1-4m\\left(1-\\frac{1}{m}\\left(\\frac{\\Gamma(m+\\frac{1}{2})}{\\Gamma(m)}\\right)^2\\right)\\right)}{2m\\left(1-\\frac{1}{m}\\left(\\frac{\\Gamma(m+\\frac{1}{2})}{\\Gamma(m)}\\right)^2\\right)^{3/2}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}=3+\\frac{-6\\left(\\frac{\\Gamma(m+\\frac{1}{2})}{\\Gamma(m)\\sqrt{m}}\\right)^{4}m+\\left(8m-2\\right)\\left(\\frac{\\Gamma(m+\\frac{1}{2})}{\\Gamma(m)\\sqrt{m}}\\right)^{2}-2m+1}{m\\left(1-\\frac{1}{m}\\left(\\frac{\\Gamma(m+\\frac{1}{2})}{\\Gamma(m)}\\right)^2\\right)^{2}}\\\\
\\mathrm{Median}(X)=\\sqrt{\\frac{\\Omega}{m}P^{-1}\\left(m,\\frac{1}{2}\\right)}\\\\
\\mathrm{Mode}(X)=\\frac{\\sqrt{2}}{2}\\left(\\frac{(2m-1)\\Omega}{m}\\right)^{1/2}\\\\
u:\\text{Uniform[0,1] random varible}\\\\
P\\left(a,x\\right)=\\frac{\\gamma(a,x)}{\\Gamma(a)}:\\text{Regularized lower incomplete gamma function}\\\\
P^{-1}\\left(a,u\\right):\\text{Inverse of regularized lower incomplete gamma function}\\\\

NC_CHI_SQUARE
X\\sim\\mathrm{NoncentralChiSquare}\\left(\\lambda,n\\right)\\\\
x\\in [0,+\\infty)\\\\
\\lambda\\in\\mathbb{R}^{+},n\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=1-Q_{\\frac{n}{2}}\\left(\\sqrt{\\lambda},\\sqrt{x}\\right)\\\\
f_{X}\\left(x\\right)=\\frac{1}{2}e^{-(x+\\lambda)/2}\\left (\\frac{x}{\\lambda}\\right)^{n/4-1/2}I_{n/2-1}(\\sqrt{\\lambda x})\\\\
Sample_{X}=\\sum_{i=1}^{n}\\left(\\sqrt{\\frac{\\lambda}{n}}+\\Phi^{-1}\\left(u_{i}\\right)\\right)^{2}\\\\
\\mu'_{k}=E[X^k]=\\int_{0}^{\\infty}x^{k}f_{X}\\left(x\\right)dx=2^{k-1}(k-1)!(n+k\\lambda)+\\sum_{j=1}^{k-1}\\frac{(k-1)!2^{j-1}}{(k-j)!}(n+j\\lambda )\\mu'_{k-j}\\\\
\\mathrm{Mean}(X)=\\mu'_{1}=n+\\lambda\\\\
\\mathrm{Variance}(X)=\\mu'_{2}-\\mu'^{2}_{1}=2(n+2\\lambda)\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}=\\frac{2^{3/2}(n+3\\lambda)}{(n+2\\lambda)^{3/2}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}=\\frac{12(n+4\\lambda)}{(n+2\\lambda)^2}\\\\
\\mathrm{Median}(X)=F^{-1}_{X}\\left(\\frac{1}{2}\\right)\\\\
\\mathrm{Mode}(X)=\\arg\\max_{x}f_{X}\\left(x\\right)\\\\
\\text{It is not possible to compute an analytic function for the inverse of the cumulative}\\\\
\\text{density function. However,we can compute a random sample from the distribution.}\\\\
u_{i}:\\text{Uniform[0,1] random varible}\\\\
\\Phi^{-1}\\left(x\\right):\\text{Inverse of cumulative function from standard normal distribution}\\\\
Q_M(a,b): \\text{Marcum function of order }M\\in\\mathbb{N}\\\\
I_{\\alpha}\\left(x\\right):\\text{Modified Bessel function of the first kind of order }\\alpha\\in\\mathbb{N}\\\\

NC_F
X\\sim\\mathrm{NonCentralF}\\left(\\lambda,n_{1},n_{2}\\right)\\\\
x\\in\\left[0,\\infty\\right)\\\\
\\lambda\\in\\mathbb{R}^{+},n_{1}\\in\\mathbb{R}^{+},n_{2}\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=\\sum\\limits_{j=0}^\\infty\\left(\\frac{\\left(\\frac{1}{2}\\lambda\\right)^j}{j!}e^{-\\lambda/2}\\right)I_{n_1x/(n_2+n_1x)}\\left(\\frac{n_1}{2}+j,\\frac{n_2}{2}\\right)\\\\
f_{X}\\left(x\\right)=\\sum\\limits_{k=0}^\\infty\\frac{e^{-\\lambda/2}(\\lambda/2)^k}{ Beta\\left(\\frac{n_2}{2},\\frac{n_1}{2}+k\\right) k!}\\left(\\frac{n_1}{n_2}\\right)^{\\frac{n_1}{2}+k}\\left(\\frac{n_2}{n_2+n_1x}\\right)^{\\frac{n_1+n_2}{2}+k}x^{n_1/2-1+k}\\\\
Sample_{X}=\\frac{\\left(\\sum_{i=1}^{n_1}\\left(\\sqrt{\\frac{\\lambda}{n_1}}+\\Phi^{-1}\\left(u_{i}\\right)\\right)^{2}\\right)/n_1}{\\left(2P^{-1}\\left(\\frac{n_2}{2},u\\right)\\right)/n_2}\\\\
\\mu'_{k}=E[X^k]=\\int_{0}^{\\infty}x^{k}f_{X}\\left(x\\right)dx=e^{-\\lambda/2}\\left(\\frac{n1}{n2}\\right)^{k}\\frac{\\Gamma\\left(n_1/2-k\\right)}{\\Gamma\\left(n_1/2\\right)}\\sum_{r=0}^{\\infty }\\left(\\frac{1}{r!}\\right)\\left(\\frac{\\lambda}{2}\\right)^{r}\\frac{\\Gamma\\left(\\frac{m}{2}+r+k\\right)}{\\Gamma\\left(\\frac{m}{2}+r\\right)}\\\\
\\mathrm{Mean}(X)=\\mu'_{1}\\\\
\\mathrm{Variance}(X)=\\mu'_{2}-\\mu'^{2}_{1}\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}\\\\
\\mathrm{Median}(X)=F^{-1}_{X}\\left(\\frac{1}{2}\\right)\\\\
\\mathrm{Mode}(X)=\\arg\\max_{x}f_{X}\\left(x\\right)\\\\
\\text{It is not possible to compute an analytic function for the inverse of the cumulative}\\\\
\\text{density function. However,we can compute a random sample from the distribution.}\\\\
u:\\text{Uniform[0,1] random varible}\\\\
u_{i}:\\text{Uniform[0,1] random varible}\\\\
\\Phi^{-1}\\left(x\\right):\\text{Inverse of cumulative function from standard normal distribution}\\\\
P^{-1}\\left(a,u\\right):\\text{Inverse of regularized lower incomplete gamma function}\\\\
Beta\\left(x,y\\right):\\text{Beta function}\\\\

NC_T_STUDENT
X\\sim\\mathrm{NonCentralTStudent}\\left(\\lambda,n,L,S\\right)\\\\
x\\in\\left(-\\infty,\\infty\\right)\\\\
\\lambda\\in\\mathbb{R},n\\in\\mathbb{R}^{+},S\\in\\mathbb{R}^{+},L\\in\\mathbb{R}\\\\
F_{X}\\left(x\\right)=\\left\\{\\begin{array}{cl}\\frac{1}{2}\\sum_{j=0}^\\infty\\frac{1}{j!}(-\\lambda\\sqrt{2})^je^{\\frac{-\\lambda^2}{2}}\\frac{\\Gamma(\\frac{j+1}{2})}{\\sqrt{\\pi}}I_{n/(n+z(x)^2)}\\left (\\frac{n}{2},\\frac{j+1}{2}\\right ) & \\text{if } \\ z(x)\\ge 0 \\\\ 1-\\frac{1}{2}\\sum_{j=0}^\\infty\\frac{1}{j!}(-\\lambda\\sqrt{2})^je^{\\frac{-\\lambda^2}{2}}\\frac{\\Gamma(\\frac{j+1}{2})}{\\sqrt{\\pi}}I_{n/(n+z(x)^2)}\\left (\\frac{n}{2},\\frac{j+1}{2}\\right ) & \\text{if } \\ z(x) < 0\\end{array} \\right.\\\\
f_{X}\\left(x\\right)=\\frac{1}{S}\\frac{n^{n/2}\\Gamma\\left(n+1\\right)}{2^{n}e^{\\lambda^{2}/2}\\left(n+z(x)^{2}\\right)^{n/2}\\Gamma\\left(n/2\\right)}\\times \\\\ \\left\\{ \\frac{\\sqrt{2}\\lambda z(x)_{1}F_{1}\\left(\\frac{n}{2}+1,\\frac{3}{2},\\frac{\\lambda^{2}z(x)^{2}}{2\\left(n+z(x)^{2}\\right)}\\right)}{\\left(n+z(x)^{2}\\right)\\Gamma\\left(\\frac{n+1}{2}\\right)} - \\frac{_{1}F_{1}\\left(\\frac{n+1}{2},\\frac{1}{2},\\frac{\\lambda^{2}z(x)^{2}}{2\\left(n+z(x)^{2}\\right)}\\right)}{\\sqrt{n+z(x)^{2}}\\Gamma\\left(\\frac{n}{2}+1\\right)} \\right\\}\\\\
Sample_{X}=L+S\\frac{\\left(\\lambda+\\Phi^{-1}\\left(u\\right)\\right)}{\\left(\\sqrt{2P^{-1}\\left(\\frac{n}{2},u\\right)}\\right)/n}\\\\
\\tilde{\\mu}'_{k}=E[\\tilde{X}^k]=\\int_{0}^{\\infty}x^{k}f_{\\tilde{X}}\\left(x\\right)dx=\\frac{e^{-\\lambda^{2}/2}}{\\sqrt{n\\pi}\\Gamma\\left(n/2\\right)}\\Gamma\\left(\\frac{n-k}{2}\\right)n^{k/2}\\sum_{r=0}^{\\infty }\\frac{\\lambda^{r}2^{r/2}}{r!}\\Gamma\\left(\\frac{r+k+1}{2}\\right)\\\\
\\mathrm{Mean}(X)=L+S\\tilde{\\mu}'_{1}\\\\
\\mathrm{Variance}(X)=S^{2}(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})\\\\
\\mathrm{Skewness}(X)=\\frac{\\tilde{\\mu}'_{3}-3\\tilde{\\mu}'_{2}\\tilde{\\mu}'_{1}+2\\tilde{\\mu}'^{3}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{1.5}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\tilde{\\mu}'_{4}-4\\tilde{\\mu}'_{1}\\tilde{\\mu}'_{3}+6\\tilde{\\mu}'^{2}_{1}\\tilde{\\mu}'_{2}-3\\tilde{\\mu}'^{4}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{2}}\\\\
\\mathrm{Median}(X)=F^{-1}_{X}\\left(\\frac{1}{2}\\right)\\\\
\\mathrm{Mode}(X)=\\arg\\max_{x}f_{X}\\left(x\\right)\\\\
\\tilde{X}\\sim\\mathrm{NonCentralTStudent}\\left(\\lambda,n,0,1\\right)\\\\
\\text{It is not possible to compute an analytic function for the inverse of the cumulative}\\\\
\\text{density function. However,we can compute a random sample from the distribution.}\\\\
L:\\text{Location parameter}\\\\
S:\\text{Scale parameter}\\\\
z\\left(x\\right)=\\left(x-L\\right)/S\\\\
u:\\text{Uniform[0,1] random varible}\\\\
P^{-1}\\left(a,u\\right):\\text{Inverse of regularized lower incomplete gamma function}\\\\
I_{\\alpha}\\left(x\\right):\\text{Modified Bessel function of the first kind of order }\\alpha\\in\\mathbb{N}\\\\
_{1}F_{1}(a,b,z):\\text{Kummer's confluent hypergeometric function}\\\\

NORMAL
X\\sim\\mathrm{Normal}\\left(\\mu,\\sigma\\right)\\\\
x\\in\\left(-\\infty,\\infty\\right)\\\\
\\mu\\in\\mathbb{R},\\sigma\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=\\frac{1}{2}\\left[1+\\operatorname{erf}\\left(\\frac{x-\\mu}{\\sigma\\sqrt{2}}\\right)\\right]=\\Phi\\left(\\frac{x-\\mu}{\\sigma}\\right)\\\\
f_{X}\\left(x\\right)=\\frac{1}{\\sigma\\sqrt{2\\pi}} e^{-\\frac{1}{2}\\left(\\frac{x-\\mu}{\\sigma}\\right)^2}=\\phi\\left(\\frac{x-\\mu}{\\sigma}\\right)\\\\
F^{-1}_{X}\\left(u\\right)=\\mu+\\sigma\\sqrt{2} \\operatorname{erf}^{-1}(2u-1)=\\mu+\\sigma\\Phi^{-1}\\left(u\\right)\\\\
\\mu'_{k}=E[X^k]=\\int_{-\\infty}^{\\infty}x^{k}f_{X}\\left(x\\right)dx=\\sigma^k\\cdot (-i\\sqrt 2)^k U\\left(-\\frac{k}{2},\\frac{1}{2},-\\frac{1}{2}\\left(\\frac \\mu \\sigma\\right)^2\\right)\\\\
\\mathrm{Mean}(X)=\\mu'_{1}=\\mu\\\\
\\mathrm{Variance}(X)=\\mu'_{2}-\\mu'^{2}_{1}=\\sigma^{2}\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}=0\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}=3\\\\
\\mathrm{Median}(X)=\\mu\\\\
\\mathrm{Mode}(X)=\\mu\\\\
\\mu:\\text{Location parameter}\\\\
\\sigma:\\text{Scale parameter}\\\\
u:\\text{Uniform[0,1] random varible}\\\\
U(a,b,z):\\text{Tricomi's confluent hypergeometric function}\\\\
\\Phi\\left(x\\right):\\text{Cumulative function from standard normal distribution}\\\\
\\Phi^{-1}\\left(x\\right):\\text{Inverse of cumulative function from standard normal distribution}\\\\
\\phi\\left(x\\right):\\text{Density function from standard normal distribution}\\\\
\\mathrm{erf}(x):\\text{Error function}\\\\
\\mathrm{erf}^{-1}(x):\\text{Inverse of error function}\\\\

PARETO_FIRST_KIND
X\\sim\\mathrm{ParetoFirstKind}\\left(x_\\mathrm{m},\\alpha,L\\right)\\\\
x\\in [L+x_\\mathrm{m},\\infty)\\\\
x_\\mathrm{m}\\in\\mathbb{R}^{+},\\alpha\\in\\mathbb{R}^{+},L\\in\\mathbb{R}\\\\
F_{X}\\left(x\\right)=1-\\left(\\frac{x_\\mathrm{m}}{x-L}\\right)^\\alpha\\\\
f_{X}\\left(x\\right)=\\frac{\\alpha x_\\mathrm{m}^\\alpha}{(x-L)^{\\alpha+1}}\\\\
F^{-1}_{X}\\left(u\\right)=L+x_\\mathrm{m} {(1-u)}^{-\\frac{1}{\\alpha}}\\\\
\\tilde{\\mu}'_{k}=E[\\tilde{X}^k]=\\int_{x_\\mathrm{m}}^{\\infty}x^{k}f_{\\tilde{X}}\\left(x\\right)dx=\\left\\{\\begin{array}{cl}\\infty & \\text{if } \\alpha\\le k \\\\ \\frac{\\alpha x_\\mathrm{m}^k}{\\alpha-k} & \\text{if } \\alpha>k \\end{array} \\right.\\\\
\\mathrm{Mean}(X)=L+\\tilde{\\mu}'_{1}=L+\\dfrac{\\alpha x_\\mathrm{m}}{\\alpha-1} \\quad \\text{if }\\alpha>1\\\\
\\mathrm{Variance}(X)=(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})=\\dfrac{x_\\mathrm{m}^2\\alpha}{(\\alpha- 1)^2(\\alpha-2)} \\quad \\text{if }\\alpha>2\\\\
\\mathrm{Skewness}(X)=\\frac{\\tilde{\\mu}'_{3}-3\\tilde{\\mu}'_{2}\\tilde{\\mu}'_{1}+2\\tilde{\\mu}'^{3}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{1.5}}=\\frac{2(1+\\alpha)}{\\alpha-3}\\sqrt{\\frac{\\alpha-2}{\\alpha}}\\quad \\text{if } \\alpha>3\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\tilde{\\mu}'_{4}-4\\tilde{\\mu}'_{1}\\tilde{\\mu}'_{3}+6\\tilde{\\mu}'^{2}_{1}\\tilde{\\mu}'_{2}-3\\tilde{\\mu}'^{4}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{2}}=\\frac{6(\\alpha^3+\\alpha^2-6\\alpha-2)}{\\alpha(\\alpha-3)(\\alpha-4)}\\quad \\text{if }\\alpha>4\\\\
\\mathrm{Median}(X)=L+x_\\mathrm{m} \\sqrt[\\alpha]{2}\\\\
\\mathrm{Mode}(X)=L+x_\\mathrm{m}\\\\
\\tilde{X}\\sim\\mathrm{ParetoFirstKind}\\left(x_\\mathrm{m},\\alpha,0\\right)\\\\
L:\\text{Location parameter}\\\\
x_\\mathrm{m}:\\text{Scale parameter}\\\\
u:\\text{Uniform[0,1] random varible}\\\\

PARETO_SECOND_KIND
X\\sim\\mathrm{ParetoSecondKind}\\left(x_\\mathrm{m},\\alpha,L\\right)\\\\
x\\in\\left(L,\\infty\\right)\\\\
x_\\mathrm{m}\\in\\mathbb{R}^{+},\\alpha\\in\\mathbb{R}^{+},L\\in\\mathbb{R}\\\\
F_{X}\\left(x\\right)=1-\\left[{1+{x-L \\over x_\\mathrm{m}}}\\right]^{-\\alpha}\\\\
f_{X}\\left(x\\right)={\\alpha \\over x_\\mathrm{m}} \\left[{1+{x-L \\over x_\\mathrm{m}}}\\right]^{-(\\alpha+1)}\\\\
F^{-1}_{X}\\left(u\\right)=L+x_\\mathrm{m} \\left[\\left(1-p\\right)^{-\\frac{1}{\\alpha}} -1\\right]\\\\
\\tilde{\\mu}'_{k}=E[\\tilde{X}^k]=\\int_{0}^{\\infty}x^{k}f_{\\tilde{X}}\\left(x\\right)dx=\\frac{x_\\mathrm{m}^k \\Gamma(\\alpha-k)\\Gamma(1+k)}{\\Gamma(\\alpha)}\\\\
\\mathrm{Mean}(X)=\\tilde{\\mu}'_{1}={x_\\mathrm{m} \\over {\\alpha -1}}  \\quad \\text{if }\\alpha>1\\\\
\\mathrm{Variance}(X)=\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1}={{x_\\mathrm{m}^2 \\alpha} \\over {(\\alpha-1)^2(\\alpha-2)}} \\quad \\text{if }\\alpha>2\\\\
\\mathrm{Skewness}(X)=\\frac{\\tilde{\\mu}'_{3}-3\\tilde{\\mu}'_{2}\\tilde{\\mu}'_{1}+2\\tilde{\\mu}'^{3}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{1.5}}=\\frac{2(1+\\alpha)}{\\alpha-3}\\,\\sqrt{\\frac{\\alpha-2}{\\alpha}} \\quad \\text{if }\\alpha>3\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\tilde{\\mu}'_{4}-4\\tilde{\\mu}'_{1}\\tilde{\\mu}'_{3}+6\\tilde{\\mu}'^{2}_{1}\\tilde{\\mu}'_{2}-3\\tilde{\\mu}'^{4}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{2}}=\\frac{6(\\alpha^3+\\alpha^2-6\\alpha-2)}{\\alpha(\\alpha-3)(\\alpha-4)} \\quad \\text{if }\\alpha>4\\\\
\\mathrm{Median}(X)=x_\\mathrm{m}\\left(\\sqrt[\\alpha]{2}-1\\right)\\\\
\\mathrm{Mode}(X)=0\\\\
X\\sim\\mathrm{ParetoSecondKind}\\left(x_\\mathrm{m},\\alpha,0\\right)\\\\
x_\\mathrm{m}:\\text{Scale parameter}\\\\
u:\\text{Uniform[0,1] random varible}\\\\
\\Gamma\\left(x\\right):\\text{Gamma function}\\\\

PERT
X\\sim\\mathrm{Pert}\\left(a,b,c\\right)\\\\
x\\in\\left[a,c\\right]\\\\
a\\in\\mathbb{R},b\\in\\mathbb{R},c\\in\\mathbb{R},a < b < c\\\\
F_{X}\\left(x\\right)=I(z(x),\\alpha_{1},\\alpha_{2})\\\\
f_{X}\\left(x\\right)=\\frac{(x-a)^{\\alpha-1}(c-x)^{\\beta-1}} {Beta(\\alpha_{1},\\alpha_{2})(c-a)^{\\alpha+\\beta-1}}\\\\
F^{-1}_{X}\\left(u\\right)=a+(c-a)\\cdot I^{-1}\\left(u,\\alpha_{1},\\alpha_{2}\\right)\\\\
\\mu'_{k}=E[X^k]=\\int_{a}^{c}x^{k}f_{X}\\left(x\\right)dx\\\\
\\mathrm{Mean}(X)=\\mu'_{1}=\\frac{a+4b+c}{6}\\\\
\\mathrm{Variance}(X)=\\mu'_{2}-\\mu'^{2}_{1}=\\frac{(\\mathrm{Mean}(X)-a)(c-\\mathrm{Mean}(X))}{7}\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}=\\frac{2\\,(\\beta-\\alpha)\\sqrt{\\alpha+\\beta+1}}{(\\alpha+\\beta+2)\\sqrt{\\alpha\\beta}} \\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}=\\frac{6[(\\alpha-\\beta)^2 (\\alpha +\\beta+1)-\\alpha \\beta (\\alpha+\\beta+2)]}{\\alpha \\beta (\\alpha+\\beta+2) (\\alpha+\\beta+3)}+3\\\\
\\mathrm{Median}(X)=a+(c-a)\\cdot I^{-1}\\left(\\frac{1}{2},\\alpha_{1},\\alpha_{2}\\right)\\\\
\\mathrm{Mode}(X)=b\\\\
z\\left(x\\right)=\\left(x-a\\right)/\\left(c-a\\right)\\\\
u:\\text{Uniform[0,1] random varible}\\\\
\\alpha_{1}=\\frac{4b+c-5a} {c-a},\\alpha_{2}=\\frac{5c-a-4b} {c-a}\\\\
I\\left(x,a,b\\right):\\text{Regularized incomplete beta function}\\\\
I^{-1}\\left(x,a,b\\right):\\text{Inverse of regularized incomplete beta function}\\\\
Beta\\left(x,y\\right):\\text{Beta function}\\\\

POWER_FUNCTION
X\\sim\\mathrm{PowerFunction}\\left(\\alpha,a,b\\right)\\\\
x\\in\\left[a,b\\right]\\\\
\\alpha\\in\\mathbb{R}^{+},a\\in\\mathbb{R},b\\in\\mathbb{R},a < b\\\\
F_{X}\\left(x\\right)=\\left(\\frac{x-a}{b-a}\\right)^{\\alpha}\\\\
f_{X}\\left(x\\right)=\\frac{\\alpha(x-a)^{\\alpha-1}}{(b-a)^\\alpha}\\\\
F^{-1}_{X}\\left(u\\right)=\\left[a+u(b-a)\\right]^{-\\alpha}\\\\
\\mu'_{k}=E[X^k]=\\int_{a}^{b}x^{k}f_{X}\\left(x\\right)dx\\\\
\\mathrm{Mean}(X)=\\mu'_{1}=\\frac{a+b\\alpha}{\\alpha+1}\\\\
\\mathrm{Variance}(X)=\\mu'_{2}-\\mu'^{2}_{1}=\\frac{2a^2+2ab\\alpha+b^2\\alpha(\\alpha+1)}{(\\alpha+1)(\\alpha+2)}-\\mathrm{Mean}(X)^{2}\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}=2\\left(1-\\alpha\\right)\\sqrt{\\frac{\\alpha+2}{\\alpha\\left(\\alpha+3\\right)}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}=\\frac{6\\left(\\alpha^{3}-\\alpha^{2}-6\\alpha+2\\right)}{\\alpha\\left(\\alpha+3\\right)\\left(\\alpha+4\\right)}+3\\\\
\\mathrm{Median}(X)=\\left[a+frac{1}{2}(b-a)\\right]^{-\\alpha}\\\\
\\mathrm{Mode}(X)=\\text{undefined}\\\\
a:\\text{Location parameter}\\\\
b-a:\\text{Scale parameter}\\\\
u:\\text{Uniform[0,1] random varible}\\\\

RAYLEIGH
X\\sim\\mathrm{Rayleigh}\\left(L,S\\right)\\\\
x\\in\\left[L,\\infty\\right)\\\\
L\\in\\mathbb{R},S\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=1-e^{-z(x)^{2}/2}\\\\
f_{X}\\left(x\\right)=z(x)\\times e^{-z(x)^{2}/2}/S\\\\
F^{-1}_{X}\\left(u\\right)=L+S\\sqrt{-2\\log\\left(1-u\\right)}\\\\
\\mu'_{k}=E[\\tilde{X}^k]=\\int_{0}^{\\infty}x^{k}f_{\\tilde{X}}\\left(x\\right)dx=\\sqrt{2^{k}}\\Gamma\\left(\\frac{k}{2}+1\\right)\\\\
\\mathrm{Mean}(X)=L+S\\cdot\\mu'_{1}=L+S\\sqrt{\\frac{\\pi}{2}}\\\\
\\mathrm{Variance}(X)=S^{2}(\\mu'_{2}-\\mu'^{2}_{1})=S^{2}\\frac{4-\\pi}{2}\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}=\\frac{2\\left(\\pi-3\\right)\\sqrt{\\pi}}{\\left(4-\\pi\\right)^{3/2}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}=3+\\frac{24\\pi-6\\pi^{2}-16}{\\left(4-\\pi\\right)^{2}}\\\\
\\mathrm{Median}(X)=L+S\\sqrt{-2\\log\\left(\\frac{1}{2}\\right)}\\\\
\\mathrm{Mode}(X)=L+S\\\\
\\tilde{X}\\sim\\mathrm{Rayleigh}\\left(0,1\\right)\\\\
L:\\text{Location parameter}\\\\
S:\\text{Scale parameter}\\\\
z\\left(x\\right)=\\left(x-L\\right)/S\\\\
u:\\text{Uniform[0,1] random varible}\\\\
\\Gamma\\left(x\\right):\\text{Gamma function}\\\\

RECIPROCAL
X\\sim\\mathrm{Reciprocal}\\left(a,b\\right)\\\\
x\\in\\left[a,b\\right]\\\\
a\\in\\mathbb{R}^{+},b\\in\\mathbb{R}^{+},a < b\\\\
F_{X}\\left(x\\right)=\\frac{\\ln(x)-\\ln(a)}{\\ln(b)-\\ln(a)}\\\\
f_{X}\\left(x\\right)=\\frac{1}{x\\left(\\ln(b)-\\ln(a)\\right)}\\\\
F^{-1}_{X}\\left(u\\right)=\\exp(\\ln(a)+u\\times \\left(\\ln(b)-\\ln(a)\\right))\\\\
\\mu'_{k}=E[X^k]=\\int_{a }^{b}x^{k}f_{X}\\left(x\\right)dx=\\frac{b^k-a^k}{k\\left(\\ln(b)-\\ln(a)\\right)}\\\\
\\mathrm{Mean}(X)=\\mu'_{1}\\\\
\\mathrm{Variance}(X)=\\mu'_{2}-\\mu'^{2}_{1}\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}\\\\
\\mathrm{Median}(X)=\\exp\\left[\\ln(a)+\\frac{\\left(\\ln(b)-\\ln(a)\\right)}{2}\\right]\\\\
\\mathrm{Mode}(X)=a\\\\
u:\\text{Uniform[0,1] random varible}\\\\

RICE
X\\sim\\mathrm{Rice}\\left(v,\\sigma\\right)\\\\
x\\in [0,\\infty)\\\\
v\\in\\mathbb{R}^{+},\\sigma\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=1-Q_1\\left(\\frac{v}{\\sigma},\\frac{x}{\\sigma }\\right)\\\\
f_{X}\\left(x\\right)=\\frac{x}{\\sigma^2}\\exp\\left(\\frac{-(x^2+v^2)}{2\\sigma^2}\\right)I_0\\left(\\frac{xv}{\\sigma^2}\\right)\\\\
Sample_{X}=\\sqrt{\\Phi^{-1}(u_{1},v,\\sigma)^{2}+\\Phi^{-1}(u_{2},0,\\sigma)^{2}}\\\\
\\mu'_{k}=E[X^k]=\\int_{-\\infty }^{\\infty }x^{k}f_{X}\\left(x\\right)dx=\\sigma^k2^{k/2}\\,\\Gamma(1+k/2)\\,L_{k/2}(-v^2/2\\sigma^2)\\\\
\\mathrm{Mean}(X)=\\mu'_{1}\\\\
\\mathrm{Variance}(X)=\\mu'_{2}-\\mu'^{2}_{1}\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}\\\\
\\mathrm{Median}(X)=F^{-1}_{X}\\left(\\frac{1}{2}\\right)\\\\
\\mathrm{Mode}(X)=\\arg\\max_{x}f_{X}\\left(x\\right)\\\\
\\text{It is not possible to compute an analytic function for the inverse of the cumulative}\\\\
\\text{density function. However,we can compute a random sample from the distribution.}\\\\
\\Phi^{-1}\\left(u,mean,variance\\right):\\text{Inverse of cumulative function from normal distribution}\\\\
L_{r}\\left(x\\right): \\text{Laguerre polynomials of order }r\\in\\mathbb{R}\\\\
L_{\\frac{1}{2}}\\left(x\\right)=e^{x/2} (x) I_{1}\\left(\\frac{x}{2}\\right)-e^{x/2} (x-1) I_{0}\\left(\\frac{x}{2}\\right)\\\\
L_{\\frac{3}{2}}\\left(x\\right)=\\frac{1}{3} e^{x/2} (2 x^2-6 x+3) I_0(x/2)-\\frac{2}{3} e^{x/2} (x-2) x I_1(x/2)\\\\
I_{\\alpha}\\left(x\\right): \\text{Modified Bessel function of the first kind of order }\\alpha\\in\\mathbb{N}\\\\
Q_{k}(a,b): \\text{Marcum Q-function of order k }\\in\\mathbb{N}\\\\
u_{1}:\\text{Uniform[0,1] random varible}\\\\
u_{2}:\\text{Uniform[0,1] random varible}\\\\

SEMICIRCULAR
X\\sim\\mathrm{Semicircular}\\left(L,R\\right)\\\\
x\\in\\left[L,\\infty\\right)\\\\
L\\in\\mathbb{R},R\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=\\frac12+\\frac{z(x)\\sqrt{R^2-z(x)^2}}{\\pi R^2}+\\frac{\\arcsin\\!\\left(\\frac{z(x)}{R}\\right)}{\\pi}\\\\
f_{X}\\left(x\\right)=\\frac2{\\pi R^2}\\,\\sqrt{R^2-z(x)^2}\\\\
F^{-1}_{X}\\left(u\\right)=L+R\\times (2I^{-1}\\left(u,1.5,1.5\\right)-1)\\\\
\\mu'_{k}=E[X^k]=\\int_{L}^{\\infty }x^{k}f_{X}\\left(x\\right)dx\\\\
\\mathrm{Mean}(X)=\\mu'_{1}=L\\\\
\\mathrm{Variance}(X)=\\mu'_{2}-\\mu'^{2}_{1}=\\frac{R^2}{4}\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}=0\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}=2\\\\
\\mathrm{Median}(X)=L\\\\
\\mathrm{Mode}(X)=L\\\\
L:\\text{Location parameter}\\\\
R:\\text{Scale parameter}\\\\
z\\left(x\\right)= x-L\\\\
u:\\text{Uniform[0,1] random varible}\\\\

T_STUDENT
X\\sim\\mathrm{TStudent}\\left(df\\right)\\\\ 
x\\in\\left(-\\infty,\\infty\\right)\\\\
df\\in\\mathbb{R}^{+}\\\\ 
F_{X}\\left(x\\right)=I\\left(\\frac{x+\\sqrt{x^{2}+df}}{2\\sqrt{x^{2}+df}},\\frac{df}{2},\\frac{df}{2}\\right)\\\\ 
f_{X}\\left(x\\right)=\\frac{\\left(1+x^{2}/df\\right)^{-(1+df)/2}}{\\sqrt{df}\\times Beta\\left(\\frac{1}{2},\\frac{df}{2}\\right)}\\\\ 
F^{-1}_{X}\\left(u\\right)=\\left\\{\\begin{array}{cl} \\sqrt{\\frac{df(1-I^{-1}\\left(u,df/2,df/2\\right))}{I^{-1}\\left(u,df/2,df/2\\right)}} & \\text{if } \\ u \\geq frac{1}{2} \\\\ -\\sqrt{\\frac{df(1-I^{-1}\\left(u,df/2,df/2\\right))}{I^{-1}\\left(u,df/2,df/2\\right)}} & \\text{if } \\ u < frac{1}{2} \\end{array} \\right.\\\\ 
\\mu'_{k}=E[X^k]=\\int_{-\\infty }^{\\infty }x^{k}f_{X}\\left(x\\right)dx=\\left\\{\\begin{array}{cl} 0 & \\text{if } \\ k\\text{ odd} \\ \\wedge \\ 0 < k < df \\\\ df^{\\frac{k}{2}} \\,\\prod_{i=1}^{k/2}\\frac{2i-1}{df-2i} & \\text{if } \\ k\\text{ even} \\ \\wedge \\ 0 < k < df \\end{array} \\right.\\\\ 
\\mathrm{Mean}(X)=\\mu'_{1}=0\\\\ 
\\mathrm{Variance}(X)=\\mu'_{2}-\\mu'^{2}_{1}=\\left\\{\\begin{array}{cl} df/(df+2) & \\text{if } \\ df > 2 \\\\ \\text{undefined} & \\text{if } \\  df \\leq 2 \\end{array} \\right.\\\\ 
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}=\\left\\{\\begin{array}{cl} 0 & \\text{if } \\ df > 3 \\\\ \\text{undefined} & \\text{if } \\  df \\leq 3 \\end{array} \\right.\\\\ 
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}=\\left\\{\\begin{array}{cl} 6/(df-4) & \\text{if } \\ df > 4 \\\\ \\text{undefined} & \\text{if } \\ df \\leq 4 \\end{array} \\right.\\\\ 
\\mathrm{Median}(X)=0\\\\ 
\\mathrm{Mode}(X)=0\\\\
u:\\text{Uniform[0,1] random varible}\\\\
I\\left(x,a,b\\right):\\text{Regularized incomplete beta function}\\\\
I^{-1}\\left(x,a,b\\right):\\text{Inverse of regularized incomplete beta function}\\\\
Beta\\left(x,y\\right):\\text{Beta function}\\\\

T_STUDENT_3P
X\\sim\\mathrm{TStudent_{3P}}\\left(df,L,S\\right)\\\\ 
x\\in\\left(-\\infty,\\infty\\right)\\\\
df\\in\\mathbb{R}^{+},L\\in\\mathbb{R},S\\in\\mathbb{R}^{+}\\\\ 
F_{X}\\left(x\\right)=I\\left(\\frac{z(x)+\\sqrt{z(x)^{2}+df}}{2\\sqrt{z(x)^{2}+df}},\\frac{df}{2},\\frac{df}{2}\\right)\\\\ 
f_{X}\\left(x\\right)=\\frac{\\left(1+z(x)^{2}/df\\right)^{-(1+df)/2}}{\\sqrt{df}\\times Beta\\left(\\frac{1}{2},\\frac{df}{2}\\right)}\\\\ 
F^{-1}_{X}\\left(u\\right)=\\left\\{\\begin{array}{cl} L+S \\ \\sqrt{\\frac{df(1-I^{-1}\\left(u,df/2,df/2\\right))}{I^{-1}\\left(u,df/2,df/2\\right)}} & \\text{if } \\ u \\geq frac{1}{2} \\\\ L-S \\ \\sqrt{\\frac{df(1-I^{-1}\\left(u,df/2,df/2\\right))}{I^{-1}\\left(u,df/2,df/2\\right)}} & \\text{if } \\ u < frac{1}{2} \\end{array} \\right.\\\\ 
\\tilde{\\mu}'_{k}=E[\\tilde{X}^k]=\\int_{0}^{\\infty}x^{k}f_{\\tilde{X}}\\left(x\\right)dx=\\left\\{\\begin{array}{cl} 0 & \\text{if } \\ k\\text{ odd} \\ \\wedge \\ 0 < k < df \\\\ df^{\\frac{k}{2}} \\,\\prod_{i=1}^{k/2}\\frac{2i-1}{df-2i} & \\text{if } \\ k\\text{ even} \\ \\wedge \\ 0 < k < df \\end{array} \\right.\\\\ 
\\mathrm{Mean}(X)=L+S\\cdot\\tilde{\\mu}'_{1}=L\\\\ 
\\mathrm{Variance}(X)=S^{2}\\times (\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})=\\left\\{\\begin{array}{cl} S^{2} \\ df/(df+2) & \\text{if } \\ df > 2 \\\\ \\text{undefined} & \\text{if } \\  df \\leq 2 \\end{array} \\right.\\\\ 
\\mathrm{Skewness}(X)=\\frac{\\tilde{\\mu}'_{3}-3\\tilde{\\mu}'_{2}\\tilde{\\mu}'_{1}+2\\tilde{\\mu}'^{3}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{1.5}}=\\left\\{\\begin{array}{cl} 0 & \\text{if } \\ df > 3 \\\\ \\text{undefined} & \\text{if } \\  df \\leq 3 \\end{array} \\right.\\\\ 
\\mathrm{Kurtosis}(X)=\\frac{\\tilde{\\mu}'_{4}-4\\tilde{\\mu}'_{1}\\tilde{\\mu}'_{3}+6\\tilde{\\mu}'^{2}_{1}\\tilde{\\mu}'_{2}-3\\tilde{\\mu}'^{4}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{2}}=\\left\\{\\begin{array}{cl} 6/(df-4) & \\text{if } \\ df > 4 \\\\ \\text{undefined} & \\text{if } \\ df \\leq 4 \\end{array} \\right.\\\\ 
\\mathrm{Median}(X)=L\\\\ 
\\mathrm{Mode}(X)=L\\\\
\\tilde{X}\\sim\\mathrm{TStudent}\\left(df\\right)\\\\
L:\\text{Location parameter}\\\\
S:\\text{Scale parameter}\\\\
z\\left(x\\right)=\\left(x-L\\right)/S\\\\
u:\\text{Uniform[0,1] random varible}\\\\
I\\left(x,a,b\\right):\\text{Regularized incomplete beta function}\\\\
I^{-1}\\left(x,a,b\\right):\\text{Inverse of regularized incomplete beta function}\\\\
Beta\\left(x,y\\right):\\text{Beta function}\\\\

TRAPEZOIDAL
X\\sim\\mathrm{Trapezoidal}\\left(a,b,c,d\\right)\\\\
x\\in\\left[a,d\\right]\\\\
a\\in\\mathbb{R},b\\in\\mathbb{R},c\\in\\mathbb{R},d\\in\\mathbb{R},a < b < c,b < c <  d\\\\
F_{X}\\left(x\\right)=\\left\\{\\begin{array}{cl}\\frac{1}{d+c-a-b}\\frac{1}{b-a}(x-a)^2 & \\text{if } \\ a\\leq x < b \\\\ \\frac{1}{d+c-a-b}(2x-a-b) & \\text{if } \\ b\\leq x < c \\\\ 1-\\frac{1}{d+c-a-b}\\frac{1}{d-c}(d-x)^2 & \\text{if } \\ c\\leq x \\le d \\end{array} \\right.\\\\
f_{X}\\left(x\\right)=\\left\\{\\begin{array}{cl}\\frac{2}{d+c-a-b}\\frac{x-a}{b-a} & \\text{if } \\ a\\leq x < b \\\\ \\frac{2}{d+c-a-b} & \\text{if } \\ b\\leq x < c \\\\ \\frac{2}{d+c-a-b}\\frac{d-x}{d-c} & \\text{if } \\ c\\leq x \\leq d \\end{array} \\right.\\\\
F^{-1}_{X}\\left(u\\right)=\\left\\{\\begin{array}{cl} a+\\sqrt{u\\times (d+c-a-b)\\times (b-a)} & \\text{if } u \\leq A_{1} \\\\ (a+b+u\\times (d+c-a-b))/2 & \\text{if } A_{1} \\leq u \\leq A_{1}+A_{2} \\\\ d-\\sqrt{(1-u)\\times (d+c-a-b)\\times (d-c)} & \\text{if } A_{1}+A_{2} \\leq u \\leq A_{1}+A_{2}+A_{3}  \\end{array} \\right.\\\\
\\mu'_{k}=E[X^k]=\\int_{a}^{b}x^{k}f_{X}\\left(x\\right)dx=\\frac{2}{d+c-b-a}\\frac{1}{(k+1)(k+2)}\\left(\\frac{d^{k+2}-c^{k+2}}{d-c}-\\frac{b^{k+2}-a^{k+2}}{b-a}\\right)\\\\
\\mathrm{Mean}(X)=\\mu'_{1}\\\\
\\mathrm{Variance}(X)=\\mu'_{2}-\\mu'^{2}_{1}\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}\\\\
\\mathrm{Median}(X)=F^{-1}_{X}\\left(1/2\\right)\\\\
\\mathrm{Mode}(X)\\in [b,c]\\\\
u:\\text{Uniform[0,1] random varible}\\\\
A_{1}=(b-a)/(d+c-a-b)\\\\
A_{2}=2(c-b)/(d+c-a-b)\\\\
A_{3}=(d-c)/(d+c-a-b)\\\\

TRIANGULAR
X\\sim\\mathrm{Triangular}\\left(a,b,c\\right)\\\\
x\\in\\left[a,b\\right]\\\\
a\\in\\mathbb{R},b\\in\\mathbb{R},c\\in\\mathbb{R},a < c < b\\\\
F_{X}\\left(x\\right)=\\left\\{\\begin{array}{cl}\\frac{(x-a)^2}{(b-a)(c-a)} & \\text{if } \\ a < x \\leq c \\\\ 1-\\frac{(b-x)^2}{(b-a)(b-c)} & \\text{if } \\ c < x < b \\\\ \\end{array} \\right.\\\\
f_{X}\\left(x\\right)=\\left\\{\\begin{array}{cl}\\frac{2(x-a)}{(b-a)(c-a)} & \\text{if } \\ a \\leq x < c,\\\\ \\frac{2(b-x)}{(b-a)(b-c)} & \\text{if } \\ c \\leq x \\leq b,\\\\ \\end{array} \\right.\\\\
F^{-1}_{X}\\left(u\\right)=\\left\\{\\begin{array}{cl} a+\\sqrt{U(b-a)(c-a)} & \\text{if } \\ 0 < U < \\frac{c-a}{b-a} \\\\ b-\\sqrt{(1-U)(b-a)(b-c)} & \\text{if } \\ \\frac{c-a}{b-a} \\leq U < 1 \\end{array} \\right.\\\\
\\mu'_{k}=E[X^k]=\\int_{a}^{b}x^{k}f_{X}\\left(x\\right)dx\\\\
\\mathrm{Mean}(X)=\\mu'_{1}=\\frac{a+b+c}{3}\\\\
\\mathrm{Variance}(X)=\\mu'_{2}-\\mu'^{2}_{1}=\\frac{a^2+b^2+c^2-ab-ac-bc}{18}\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}=\\frac{\\sqrt 2 (a\\!+\\!b\\!-\\!2c)(2a\\!-\\!b\\!-\\!c)(a\\!-\\!2b\\!+\\!c)}{5(a^2\\!+\\!b^2\\!+\\!c^2\\!-\\!ab\\!-\\!ac\\!-\\!bc)^\\frac{3}{2}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}=3-\\frac{3}{5}\\\\
\\mathrm{Median}(X)=\\left\\{\\begin{array}{cl} a+\\sqrt{\\frac{(b-a)(c-a)}{2}} & \\text{if } c \\ge\\frac{a+b}{2} \\\\ b-\\sqrt{\\frac{(b-a)(b-c)}{2}} & \\text{if } c \\le\\frac{a+b}{2} \\end{array} \\right.\\\\
\\mathrm{Mode}(X)\\in [b,c]\\\\
u:\\text{Uniform[0,1] random varible}\\\\

UNIFORM
X\\sim\\mathrm{Uniform}\\left(a,b\\right)\\\\
x\\in [a,b]\\\\
a\\in\\mathbb{R},b\\in\\mathbb{R},a < b\\\\
F_{X}\\left(x\\right)=\\frac{x-a}{b-a}\\\\
f_{X}\\left(x\\right)=\\frac{1}{b-a}\\\\
F^{-1}_{X}\\left(u\\right)=a+u\\cdot(b-a)\\\\
\\mu'_{k}=E[X^k]=\\int_{-\\infty }^{\\infty }x^{k}f_{X}\\left(x\\right)dx=\\frac{1}{k+1}\\sum_{i=0}^k a^ib^{k-i}\\\\
\\mathrm{Mean}(X)=\\mu'_{1}=\\tfrac{1}{2}(a+b)\\\\
\\mathrm{Variance}(X)=\\mu'_{2}-\\mu'^{2}_{1}\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}=0\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}=3-\\frac{6}{5}\\\\
\\mathrm{Median}(X)=\\tfrac{1}{2}(a+b)\\\\
\\mathrm{Mode}(X)\\in [a,b]\\\\
u:\\text{Uniform[0,1] random varible}\\\\

WEIBULL
X\\sim\\mathrm{Weibull}\\left(\\alpha,\\beta\\right)\\\\
x\\in [0,\\infty)\\\\
\\alpha\\in\\mathbb{R}^{+},\\beta\\in\\mathbb{R}^{+}\\\\
F_{X}\\left(x\\right)=1-e^{-(x/\\beta)^\\alpha}\\\\
f_{X}\\left(x\\right)=\\frac{\\alpha}{\\beta}\\left(\\frac{x}{\\beta}\\right)^{\\alpha-1}e^{-(x/\\beta)^\\alpha}\\\\
F^{-1}_{X}\\left(u\\right)=\\beta(-\\ln(1-u))^{1/\\alpha}\\\\
\\mu'_{k}=E[X^k]=\\int_{0}^{\\infty }x^{k}f_{X}\\left(x\\right)dx=\\beta^\\alpha \\Gamma\\left(1+\\frac{k}{\\alpha}\\right)\\\\
\\mathrm{Mean}(X)=\\mu'_{1}=\\beta\\cdot\\Gamma(1+1/\\alpha)\\\\
\\mathrm{Variance}(X)=\\mu'_{2}-\\mu'^{2}_{1}=\\beta^2\\left[\\Gamma\\left(1+2/\\alpha\\right)-\\left(\\Gamma\\left(1+1/\\alpha\\right)\\right)^2\\right]\\\\
\\mathrm{Skewness}(X)=\\frac{\\mu'_{3}-3\\mu'_{2}\\mu'_{1}+2\\mu'^{3}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{1.5}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\mu'_{4}-4\\mu'_{1}\\mu'_{3}+6\\mu'^{2}_{1}\\mu'_{2}-3\\mu'^{4}_{1}}{(\\mu'_{2}-\\mu'^{2}_{1})^{2}}\\\\
\\mathrm{Median}(X)=\\beta(\\ln(2))^{1/\\alpha}\\\\
\\mathrm{Mode}(X)=\\left\\{\\begin{array}{cl} \\beta\\left(\\frac{\\alpha-1}{\\alpha}\\right)^{1/\\alpha} & \\text{if }\\alpha>1\\\\ 0 & \\text{if } \\alpha\\leq 1 \\end{array} \\right.\\\\
\\beta:\\text{Scale parameter}\\\\
u:\\text{Uniform[0,1] random varible}\\\\
\\Gamma\\left(x\\right):\\text{Gamma function}\\\\

WEIBULL_3P
X\\sim\\mathrm{Weibull_{3P}}\\left(\\alpha,\\beta,L\\right)\\\\
x\\in [L,\\infty)\\\\
\\alpha\\in\\mathbb{R}^{+},\\beta\\in\\mathbb{R}^{+},L\\in\\mathbb{R}\\\\
F_{X}\\left(x\\right)=1-e^{-z(x)^\\alpha}\\\\
f_{X}\\left(x\\right)=\\frac{\\alpha}{\\beta}z(x)^{\\alpha-1}e^{-z(x)^\\alpha}\\\\
F^{-1}_{X}\\left(u\\right)=L+\\beta(-\\ln(1-u))^{1/\\alpha}\\\\
\\tilde{\\mu}'_{k}=E[\\tilde{X}^k]=\\int_{0}^{\\infty}x^{k}f_{\\tilde{X}}\\left(x\\right)dx=\\beta^\\alpha \\Gamma\\left(1+\\frac{k}{\\alpha}\\right)\\\\
\\mathrm{Mean}(X)=L+\\tilde{\\mu}'_{1}=L+\\beta \\ \\Gamma(1+1/\\alpha)\\\\
\\mathrm{Variance}(X)=\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1}=\\beta^2\\left[\\Gamma\\left(1+2/\\alpha\\right)-\\left(\\Gamma\\left(1+1/\\alpha\\right)\\right)^2\\right]\\\\
\\mathrm{Skewness}(X)=\\frac{\\tilde{\\mu}'_{3}-3\\tilde{\\mu}'_{2}\\tilde{\\mu}'_{1}+2\\tilde{\\mu}'^{3}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{1.5}}\\\\
\\mathrm{Kurtosis}(X)=\\frac{\\tilde{\\mu}'_{4}-4\\tilde{\\mu}'_{1}\\tilde{\\mu}'_{3}+6\\tilde{\\mu}'^{2}_{1}\\tilde{\\mu}'_{2}-3\\tilde{\\mu}'^{4}_{1}}{(\\tilde{\\mu}'_{2}-\\tilde{\\mu}'^{2}_{1})^{2}}\\\\
\\mathrm{Median}(X)=L+\\beta(\\ln(2))^{1/\\alpha}\\\\
\\mathrm{Mode}(X)=L+\\left\\{\\begin{array}{cl} \\beta\\left(\\frac{\\alpha-1}{\\alpha}\\right)^{1/\\alpha} & \\text{if }\\alpha>1\\\\ 0 & \\text{if } \\alpha\\leq 1 \\end{array} \\right.\\\\
\\tilde{X}\\sim\\mathrm{Weibull}\\left(\\alpha,\\beta\\right)\\\\
L:\\text{Location parameter}\\\\
\\beta:\\text{Scale parameter}\\\\
z\\left(x\\right)=\\left(x-L\\right)/\\beta\\\\
u:\\text{Uniform[0,1] random varible}\\\\
\\Gamma\\left(x\\right):\\text{Gamma function}\\\\
`;