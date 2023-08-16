# bessel

Pure-JS implementation of Bessel functions J,Y,I,K (for the browser and nodejs).
Emphasis on correctness and performance for integer order.

The standard notation is used here:

 - `J` is the Bessel function of the first kind
 - `Y` is the Bessel function of the second kind
 - `I` is the modified Bessel function of the first kind
 - `K` is the modified Bessel function of the second kind

## Installation

With [npm](https://www.npmjs.org/package/bessel):

```bash
$ npm install bessel
```

In the browser:

```html
<script src="bessel.js"></script>
```

The browser exposes a variable `BESSEL`

The script will manipulate `module.exports` if available (e.g. in a CommonJS
`require` context).  This is not always desirable.  To prevent the behavior,
define `DO_NOT_EXPORT_BESSEL`

## Usage

In all cases, the relevant function takes two arguments (`value`, `order`).

The return value is a JS number.  `NaN` signals an error in calculation.

- `BESSEL.besselj(x, n)` computes `J_{n}(x)`

- `BESSEL.bessely(x, n)` computes `Y_{n}(x)`

- `BESSEL.besseli(x, n)` computes `I_{n}(x)`

- `BESSEL.besselk(x, n)` computes `K_{n}(x)`

For example:

```js
// var BESSEL = require('bessel'); // uncomment this line if in node
BESSEL.besselj(1.5,0)              // 0.5118276712499389
BESSEL.bessely(1.5,0)              // 0.38244892476502895
BESSEL.besseli(1.5,0)              // 1.6467232021476754
BESSEL.besselk(1.5,0)              // 0.2138055693236539

var Y = BESSEL.bessely
Y(Math.PI, 5) + Y(Math.PI, 3) - (2 * 4 / Math.PI) * Y(Math.PI, 4) // 0
```

## Testing

`make test` will run the nodejs-based test.

To generate the `excel.tsv` test cases, make a 6-column Excel Sheet:

| x | n |    `BESSELI`   |    `BESSELJ`   |    `BESSELK`   |    `BESSELY`   |
|---|---|:--------------:|:--------------:|:--------------:|:--------------:|
| x | n |`BESSELI(A1,B1)`|`BESSELJ(A1,B1)`|`BESSELK(A1,B1)`|`BESSELY(A1,B1)`|

To generate the `mma.tsv` test cases, use the Mathematica Bessel Functions:

```mathematica
(* Bessel_[value, order] *)
F[x_,n_]:={x/2,n,BesselI[n,x/2], BesselJ[n,x/2], BesselK[n,x/2], BesselY[n,x/2]}
Do[Print[ExportString[N[F[x,n],10],"csv"]], {n,1,3}, {x,1,20} ]
```

Note: Each function follows Excel semantics `(value, order)`.  Other platforms
like Mathematica reverse the argument order.

## License

Please consult the attached LICENSE file for details.  All rights not explicitly
granted by the Apache 2.0 License are reserved by the Original Author.

## Badges

[![Build Status](https://saucelabs.com/browser-matrix/bessel.svg)](https://saucelabs.com/u/bessel)

[![Build Status](https://travis-ci.org/SheetJS/bessel.svg?branch=master)](https://travis-ci.org/SheetJS/bessel)

[![Coverage Status](http://img.shields.io/coveralls/SheetJS/bessel/master.svg)](https://coveralls.io/r/SheetJS/bessel?branch=master)

[![NPM Downloads](https://img.shields.io/npm/dt/bessel.svg)](https://npmjs.org/package/bessel)

[![Dependencies Status](https://david-dm.org/sheetjs/bessel/status.svg)](https://david-dm.org/sheetjs/bessel)

[![ghit.me](https://ghit.me/badge.svg?repo=sheetjs/bessel)](https://ghit.me/repo/sheetjs/bessel)

[![Analytics](https://ga-beacon.appspot.com/UA-36810333-1/SheetJS/bessel?pixel)](https://github.com/SheetJS/bessel)
