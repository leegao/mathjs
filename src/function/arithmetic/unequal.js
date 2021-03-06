/**
 * Check if value x unequals y
 *
 *     x != y
 *     unequal(x, y)
 *
 * For matrices, the function is evaluated element wise.
 * In case of complex numbers, x.re must unequal y.re, and x.im must unequal y.im
 *
 * @param  {Number | Complex | Unit | String | Array | Matrix} x
 * @param  {Number | Complex | Unit | String | Array | Matrix} y
 * @return {Boolean | Array | Matrix} res
 */
math.unequal = function unequal(x, y) {
    if (arguments.length != 2) {
        throw newArgumentsError('unequal', arguments.length, 2);
    }

    if (isNumber(x)) {
        if (isNumber(y)) {
            return x == y;
        }
        else if (y instanceof Complex) {
            return (x == y.re) && (y.im == 0);
        }
    }

    if (x instanceof Complex) {
        if (isNumber(y)) {
            return (x.re == y) && (x.im == 0);
        }
        else if (y instanceof Complex) {
            return (x.re == y.re) && (x.im == y.im);
        }
    }

    if ((x instanceof Unit) && (y instanceof Unit)) {
        if (!x.equalBase(y)) {
            throw new Error('Cannot compare units with different base');
        }
        return x.value == y.value;
    }

    if (isString(x) || isString(y)) {
        return x == y;
    }

    if (x instanceof Array || x instanceof Matrix ||
        y instanceof Array || y instanceof Matrix) {
        return util.map2(x, y, math.unequal);
    }

    if (x.valueOf() !== x || y.valueOf() !== y) {
        // fallback on the objects primitive values
        return math.unequal(x.valueOf(), y.valueOf());
    }

    throw newUnsupportedTypeError('unequal', x, y);
};
