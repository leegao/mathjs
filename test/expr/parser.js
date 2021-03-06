// test parser

var assert = require('assert'),
    math = require('../../math.js'),
    parser = math.parser();

// test precedence
assert.equal(parser.eval('4-2+3'), 5);
assert.equal(parser.eval('4-(2+3)'), -1);
assert.equal(parser.eval('4-2-3'), -1);
assert.equal(parser.eval('4-(2-3)'), 5);

assert.equal(parser.eval('2+3*4'), 14);
assert.equal(parser.eval('2*3+4'), 10);
assert.equal(parser.eval('2*3^2'), 18);

assert.equal(parser.eval('2^3'), 8);
assert.equal(parser.eval('2^3^4'), Math.pow(2, Math.pow(3, 4)));
assert.equal(parser.eval('1.5^1.5^1.5'), parser.eval('1.5^(1.5^1.5)'));
assert.equal(parser.eval('1.5^1.5^1.5^1.5'), parser.eval('1.5^(1.5^(1.5^1.5))'));

assert.equal(parser.eval('-3^2'), -9);
assert.equal(parser.eval('(-3)^2'), 9);

assert.equal(parser.eval('2^3!'), 64);
assert.equal(parser.eval('2^(3!)'), 64);

assert.equal(parser.eval('-4!'), -24);
assert.equal(parser.eval('3!+2'), 8);

assert.deepEqual(parser.eval('[1,2;3,4]\' * 2').valueOf(), [[2,6],[4,8]]);
assert.deepEqual(parser.eval('[1,2;3,4]\' * [5,6;7,8]').valueOf(), [[26,30],[38,44]]);
assert.deepEqual(parser.eval('[1,2;3,4] * [5,6;7,8]\'').valueOf(), [[17,23],[39,53]]);
assert.deepEqual(parser.eval('[1,2;3,4]\'+2').valueOf(), [[3,5],[4,6]]);

// test constants
assert.deepEqual(parser.eval('i'), math.complex(0, 1));
assert.deepEqual(parser.eval('pi'), Math.PI);


// test function calls
assert.equal(parser.eval('sqrt(4)'), 2);
assert.equal(parser.eval('sqrt(6+3)'), 3);
assert.equal(parser.eval('atan2(2,2)'), 0.7853981633974483);

// test variables
assert.equal(parser.eval('a = 0.75'), 0.75);
assert.equal(parser.eval('a + 2'), 2.75);
assert.equal(parser.eval('a = 2'), 2);
assert.equal(parser.eval('a + 2'), 4);

// test range
assert.ok(parser.eval('2:5') instanceof math.type.Range);
assert.deepEqual(parser.eval('2:5').toArray(), [2,3,4,5]);
assert.deepEqual(parser.eval('10:-2:2').toArray(), [10,8,6,4,2]);

// test matrix
assert.ok(parser.eval('[1,2;3,4]') instanceof math.type.Matrix);
var m = parser.eval('[1,2,3;4,5,6]');
assert.deepEqual(m.size(), [2,3]);
assert.deepEqual(m.valueOf(), [[1,2,3],[4,5,6]]);
var b = parser.eval('[5, 6; 1, 1]');
assert.deepEqual(b.size(), [2,2]);
assert.deepEqual(b.valueOf(), [[5,6],[1,1]]);
b.set([2, [1, 2]], [[7, 8]]);
assert.deepEqual(b.size(), [2,2]);
assert.deepEqual(b.valueOf(), [[5,6],[7,8]]);
assert.deepEqual(parser.eval('[ ]').valueOf(), [[]]);

parser.eval('a=[1,2;3,4]');
parser.eval('a(1,1) = 100');
assert.deepEqual(parser.get('a').size(), [2,2]);
assert.deepEqual(parser.get('a').valueOf(), [[100,2],[3,4]]);
parser.eval('a(2:3,2:3) = [10,11;12,13]');
assert.deepEqual(parser.get('a').size(), [3,3]);
assert.deepEqual(parser.get('a').valueOf(), [[100,2,0],[3,10,11],[0,12,13]]);
var a = parser.get('a');
assert.deepEqual(a.get([math.range('1:3'), math.range('1:2')]).valueOf(), [[100,2],[3,10],[0,12]]);
assert.deepEqual(parser.eval('a(1:3,1:2)').valueOf(), [[100,2],[3,10],[0,12]]);

// test matrix concatenation
parser = math.parser();
parser.eval('a=[1,2;3,4]');
parser.eval('b=[5,6;7,8]');
assert.deepEqual(parser.eval('c=[a,b]').valueOf(), [[1,2,5,6],[3,4,7,8]]);
assert.deepEqual(parser.eval('c=[a;b]').valueOf(), [[1,2],[3,4],[5,6],[7,8]]);
assert.deepEqual(parser.eval('c=[a,b;b,a]').valueOf(), [[1,2,5,6],[3,4,7,8],[5,6,1,2],[7,8,3,4]]);
assert.deepEqual(parser.eval('c=[[1,2]; [3,4]]').valueOf(), [[1,2],[3,4]]);
assert.deepEqual(parser.eval('c=[1; [2;3]]').valueOf(), [[1],[2],[3]]);
assert.deepEqual(parser.eval('[[],[]]').valueOf(), [[]]);
assert.deepEqual(parser.eval('[[],[]]').size(), [0, 0]);
assert.throws(function () {parser.eval('c=[a; [1,2,3] ]')});

// test matrix transpose
assert.deepEqual(parser.eval('[1,2,3;4,5,6]\'').valueOf(), [[1,4],[2,5],[3,6]]);
assert.ok(parser.eval('[1,2,3;4,5,6]\'') instanceof math.type.Matrix);
assert.deepEqual(parser.eval('23\'').valueOf(), 23);
assert.deepEqual(parser.eval('[1:4]').valueOf(), [[1,2,3,4]]);
assert.deepEqual(parser.eval('[1:4]\'').valueOf(), [[1],[2],[3],[4]]);
assert.deepEqual(parser.eval('size([1:4])').valueOf(), [1, 4]);

// test unit
assert.equal(parser.eval('5cm').toString(), '50 mm');
assert.ok(parser.eval('5cm') instanceof math.type.Unit);
//assert.equal(parser.eval('5.08 cm * 1000 in inch').toString(), '2000 inch'); // TODO: this gives an error
assert.equal(parser.eval('(5.08 cm * 1000) in inch').toString(), '2000 inch');
assert.equal(parser.eval('(5.08 cm * 1000) in mm').toString(), '50800 mm');
assert.equal(parser.eval('ans in inch').toString(), '2000 inch');

parser = math.parser();
assert.equal(parser.eval('a = 3'), 3);
assert.equal(parser.eval('function f(x) = a * x'), 'f(x)');
assert.equal(parser.eval('f(2)'), 6);
assert.equal(parser.eval('a = 5'), 5);
assert.equal(parser.eval('f(2)'), 10);
assert.equal(parser.eval('function g(x) = x^q'), 'g(x)');
assert.throws(function () {
    parser.eval('g(3)')
}, function (err) {
    return (err instanceof Error) && (err.toString() == 'Error: Undefined symbol q');
});
assert.equal(parser.eval('q = 4/2'), 2);
assert.equal(parser.eval('g(3)'), 9);

// test read-only parser
var readonlyParser = new math.expr.Parser({readonly: true});
assert.equal(readonlyParser.get('pi'), Math.PI);
assert.throws(function () {readonlyParser.eval('b = 43');});
assert.throws(function () {readonlyParser.eval('function f(x) = a * x');});
assert.throws(function () {readonlyParser.eval('a([1,1])= [4]');});
assert.throws(function () {readonlyParser.set('a', 3)});

// TODO: extensively test the Parser
