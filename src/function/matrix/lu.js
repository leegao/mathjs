/**
 * LU decomposition in n^3/3 time, preferable to Gaussian elimination for matrix solve
 *
 *     lu(A)
 *
 * Unfortunately, this is not a stable algorithm without requiring pivoting.
 * See: lu.tex and lu.pdf for rationale
 *
 * @param {Matrix} A
 * @return {Matrix} L and U, where the lower triangular L * upper triangular U = A
 */
math.lu = function lu (A) {
    if (arguments.length != 1) {
        throw newArgumentsError('lu', arguments.length, 1);
    }

    var size = math.size(A);
    switch (size.length) {
        case 2:
            // two dimensional array
            var rows = size[0];
            var cols = size[1];
            if (rows == cols) {
                if (A instanceof Matrix) {
                    return new Matrix(
                        _lu(A.valueOf(), rows, cols)
                    );
                }
                else {
                    // return an Array
                    return _lu(A, rows, cols);
                }
            }
            else {
                throw new RangeError('Matrix must be square ' +
                    '(size: ' + math.format(size) + ')');
            }
            break;

        default:
            // non matrix
            throw new RangeError('Matrix must be two dimensional ' +
                '(size: ' + math.format(size) + ')');
    }
};

/**
 * Calculate the inverse of a square matrix
 * @param {Array[]} matrix  A square matrix
 * @param {Number} rows     Number of rows
 * @param {Number} cols     Number of columns, must equal rows
 * @return {Array[]} inv    Inverse matrix
 * @private
 */
function _lu (matrix, rows, cols){
    var r, s, f, value, temp,
        add = math.add,
        unaryminus = math.unaryminus,
        multiply = math.multiply,
        divide = math.divide, get = math.get, range = math.range;

    // suppose gaussian elimination proceeds as M_n ... M_1 A = U, we can do the following:
    /* [1  x x x
        l1 x x x
        l2 x x x
        l3 x x x]
    We can instead do the elimination step, and just set the column to
       [1 x x x
        -l1 ...
        -l2 ...
        -l3 ... ]
    */
    var A = new Matrix(matrix); // n^2 copy overhead
    for(k = 0; k < cols; k++){
        // at column k, ensure the rest of the row is divided by A_{kk}
        // if akk = 0, we're out of luck. Use plu then
        var akk = get(A,k,k);
        // Suppose in block form, we have A = [akk x; l A'], then
        // the new (n-1 by n-1) submatrix should be A' - x*l
        var x = divide(get(A, [k,range(k+1,cols)]),akk);
        var l = unaryminus(get(A, [range(k+1,rows),k]));
        var sub = multiply(l,x);
        set(A,[k,k],1);
        set(A, [k,range(k+1,cols)], x);
        set(A, [range(k+1,rows),k], l);
        set(A, [range(k+1,rows),range(k+1,cols)], add(get(A, [range(k+1,rows),range(k+1,cols)]), sub));
    }
    return A;
};
