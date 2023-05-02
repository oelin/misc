// file contains maths, statistics and formal logic constructs
// code is pretty but ucl students don't have time to write clean code


// maths section

const notes = {

	glossary: {
		vectorSpace: `
			A vector space v has the following properties: 
			* closed under addition and subtraction
			* closed under linear combinations, i.e. (au + bv) = w where w is another vector in the space.
			* there exists a zero vector such that Av(0 + v) = v and Av(0 * v) = 0 
		`,

		vectorSpan: `
			The span of a set of vectors {v0, vn} is the set of all *linear combinations* of the 
			vectors.
		`,
		linearIndependence: `
			A set of vectors are linearly independent if none of them can be written as linear
			combinations of any others.
		`,

		linearIndependenceAndDimensionality: `
			* The number of linearly independent vectors in a spanning set S is the dimension.
			* If the span of S is the vector space V, and the vectors are LI, then they are called
			  the basis of V.
		`,
		orthonormalBasis: `
			A basis where all base vectors have unit length and are orthogonal to eachother
		`,
		matrixInverse: `
			* An inverse exists iff the determinant is non-zero.
			* An inverse exists iff the row/column vectors are linearly independent.
			* An inverse exists iff the only vectors mapped to 0 is the 0 vector itself (the kernel is trivial).
		`,
		determinant: `
			Properties:
			* |AB| = |A||B|
			* |cA| = c^n|A| for an nxn matrix
			* |A.t| = |A|
			* |A| != 0 <==> A is invertable

			* Unchanged when transposed or after adding one row of a matrix to another row of the same matrix.
			* Negated when two rows are swapped
			* Scaled by a factor c if any *row* or *column* is multiplied by c.
		`,
		kernel: `
			* Kernel or nullspace of a matrix is the set of vectors mapped to 0 by the transformation.
			* Can be found through gaussian elim because the procedure does not change the kernel.
		
			* To find ker(A), solve Ax = 0
			  1) Convert A to reduced-RE-form using row ops, giving A'.
			  2) Solve A'x = 0
			  3) Answer is the span of x.
		`,
		consistency: `
			* An SLE is inconsistent iff it cannot be solved (i.e. has no solutions).
			* An SLE is consistent if it has at least one solution.
			* An SLE is under-determined when there are more unknowns than equations, e.g. a row may be all zeros.
			* An SLE is not uniquely solvable if the determinant is 0.

			In other words, if a system is uniquely solvable, then it has a *single* solution 
			and is consistent. If a system is *not* uniquely solvable, it may either be 
			inconsistent (no solutions), or nonetheless consistent with infinite solutions.
		`,
		eigenstuff: `
			* Eigenvector - v is an eigenvector of A iff Av = kv. The eigenvectors of A are those vectors
			  which are just scaled up or down after the transformation.

			* Eigenvalue - for some eigenvector of A, the eigenvalue k, is the factor by which the vector
			  has been scaled.

			* Eigenproblem: 
			  Av = kv 
 			  <===> Av = kIv 
			  <===> Av - kIv = 0
			  <===> (A - kI)v = 0
			  <===> Require v != 0, then det(A - kI) = 0
			  So solve for det(A - kI) to find the difference eigenvalues.

			* Eigendecomposition: A = V @ L @ (V^-1)

                          - where A is the matrix to be decomposed
			  - L is a diagonal square matrix containing the eigenvalues of A
			  - V is a matrix where the nth column is the nth eigenvector of A
			  - (V^-1) is the inverse of V.
		`,
		svd:`
			Singular value decomposition, a generalisation of eigendecomp to rectangular 
			matricies.

			* A = U @ S @ (V.t) where
	
			- U is an orthonomal m by m matrix
			- S is an m by n diagonal matrix of singular non-negative values
			- V is an orthonomal n by n matrix, such that V @ V.t = I.
		`,
		innerProduct: `
			* Linearity: <ax + by, z> = a<x, z> + b<y, z>
			* Positive definiteness: <x, x> > 0
		`
	}	
}

function rad(degrees) {
	return (degrees/180) * Math.PI 
}

function deg(radians) {
	return (radians/Math.PI) * 180
}

const Operations = {

	negate(x) {
		return x.negate ? x.negate() : -x
	},


	add(x, y) {
		return x.add ? x.add(y) : (x + y)
	},


	subtract(x, y) {
		return x.subtract ? x.subtract(y) : (x - y)
	},


	times(x, y) {
		return x.times ? x.times(y) : (x * y)
	},


	divide(x, y) {
		return x.divide ? x.divide(y) : (x / y)
	}
}


class Vector {

	constructor(...elements) {
		this.elements = elements
	}

	// accessors

	copy() {
		return new Vector(...this.elements)
	}


	get(index) {
		return this.elements[index] ?? 0
	}


	set(index, value) {
		this.elements[index] = value
	}

	
	// generic transformations

	map(f) {
		return new Vector(...this.elements.map(f))
	}


	filter(f) {
		return new Vector(...this.elements.filter(f))
	}


	reduce(f) {
		return this.elements.reduce(f)
	}


	unary(f) {
		return this.map(element => f(element))
	}


	binary(vector, f) {
		return this.map((element, index) => f(element, vector.get(index)))
	}


	// vector negation

	negate() {
		return this.unary(Operations.negate)
	}


	// vector addition

	add(vector) {
		return this.binary(vector, Operations.add)
	}


	// vector subtraction

	subtract(vector) {
		return this.binary(vector, Operations.subtract)
	}


	// vector multiplication

	times(vector) {
		return this.binary(vector, Operations.times)
	}


	// vector division

	divide(vector) {
		return this.binary(vector, Operations.divide)
	}


	// vector dot product

	dot(vector) {
		return this.times(vector).sum
	}


	// vector exponentiation

	pow(exponent) {
		return this.unary(x => x ** exponent)
	}


	// vector square 

	square() {
		return this.pow(2)
	}


	// vector equality

	equals(vector) {
		return this.filter((element, i) => element !== vector.get(i)).count === 0
	}

	// scalar addition

	increase(scalar) {
		return this.unary(element => element + scalar)
	}


	// scalar subtraction

	decrease(scalar) {
		return this.unary(element => element - scalar)
	}


	// scalar multiplication

	scale(scalar) {
		return this.unary(element => element * scalar)
	}


	// scalar division

	shrink(scalar) {
		return this.unary(element => element / scalar)
	}


	// sort in ascending order

	sort() {
		return new Vector(...[...this.elements].sort((a, b) => a - b))
	}


	// stringify

	toString() {
		return `(${this.elements.join(', ')})`
	}


	// statistics...


	// sum

	get sum() {
		return this.reduce(Operations.add)
	}


	// sample size

	get count() {
		return this.elements.length
	}


	// mean

	get mean() {
		return this.sum / this.count
	}


	// population variance

	get variance() {
		return this.square().mean - (this.mean ** 2)
	}


	// sample variance

	get svariance() {
		return this.decrease(this.mean).square().sum / (this.count - 1)
	}


	// maximum value

	get max() {
		return Math.max(...this.elements)
	}


	// minimum value

	get min() {
		return Math.min(...this.elements)
	}


	// range

	get range() {
		return this.max - this.min
	}


	// index of maximum value

	get argmax() {
		return this.elements.indexOf(this.max)
	}


	// index of minimum value

	get argmin() {
		return this.elements.indexOf(this.min)
	}


	// percentiles (what value are p% of the sample less than)

	percentile(p) {
		const sorted = this.sort()
		const index = (p / 100) * (this.count - 1)
		const indexInteger = Math.ceil(index) // NOTE: this might be round() for the exam
		const indexWasWholeNumber = index == indexInteger

		if (indexWasWholeNumber) {
			// use the average of the current and next position

			return (sorted.get(indexInteger) + sorted.get(indexInteger + 1)) / 2
		}

		else {
			return sorted.get(indexInteger)
		}
	}


	// the median, middle value or 50th percentile 

	get median() {
		return this.percentile(50)
	}


	// compute the sample frequency distribution

	get frequency() {
		const table = {}
		

		this.sort().elements.forEach(element => {
			if (table.hasOwnProperty(element)) {
				table[element] += 1
			} else {
				table[element] = 1
			}
		})

		return {
			keys: Object.keys(table),
			values: Object.values(table)
		}
	}


	// the mode or most common value (assumes the distribution is not bimodal)

	get mode() {
		const freqs = this.frequency
		const argmax = new Vector(...freqs.values).argmax

		return parseFloat(freqs.keys[argmax])
	}


	// gemoetry 

	get negative() {
		return this.negate()
	}

	get x() {
		return this.get(0)
	}

	get y() {
		return this.get(1) 
	}

	get z() {
		return this.get(2) 
	}

	get w() {
		return this.get(3) 
	}

	get magnitude() {
		return Math.sqrt(
			this.elements
			.map(a => a ** 2)
			.reduce((a, b) => a + b, 0)
		)
	}

	get normal() {
		return this.shrink(this.magnitude)
	}

	angleBetween(vector) {
		return Math.acos(this.dot(vector) / (this.magnitude * vector.magnitude))		
	}

	// angle from (0, 0)

	angle(n=0) {
		return this.angleBetween(Vector.unit(n))
	}

	cross(vector) {

		const A = this
		const B = vector

		const C = new Vector(
			A.get(1)*B.get(2) - A.get(2)*B.get(1),
			A.get(2)*B.get(0) - A.get(0)*B.get(2),
			A.get(0)*B.get(1) - A.get(1)*B.get(0),
		)

		return C
	}

	get s() {
		return this.toString()
	}


	static fromAngle(radians, length) {
		return new Vector(length * Math.cos(radians), length * Math.sin(radians))
	}

	static unit(n) {
		const array = Array(n).fill(0)
		array[n] = 1

		return new Vector(...array)
	}

	static ZERO(n) {
		return new Vector(...Array(n).fill(0))
	}

	// rotate a 2d vector

	rotate2d(radians) {
		return Vector.fromAngle(this.angle() + radians, this.magnitude)
	}

	// rotate 90 degrees

	get orthogonal() {
		return this.rotate2d(rad(90))
	}

	// create an orthonormal pair

	get isNormal() {
		return this.magnitude === 1
	}


	// rotate by 90 degrees and normalise

	get orthonormal() {
		if (! this.isNormal) console.log('current vector is not normalised, using this.normal')

		return this.normal.orthogonal.normal
	}

	// transpose

	get transpose() {
		return new Matrix(...this.elements.map(element => new Vector(element)))
	}

	get t() {
		return this.transpose
	}
}


function getOrthonormalBasis(v) {
	const v1 = v.normal
	const v2 = v1.orthonormal
	const v3 = v1.cross(v2)

	return [ v1, v2, v3 ]
}


function getUnitNormalToPlane(cx, cy, cz) {

	return new Vector(cx, cy, cz).normal
}


let MatrixProd = (A, B) =>
  A.map((row, i) =>
    B[0].map((_, j) =>
      row.reduce((acc, _, n) =>
        acc + A[i][n] * B[n][j], 0
      )
    )
  )

// matricies

class Matrix {

	constructor(...vectors) {
		this.vectors = vectors
		this.rows = vectors.length
		this.columns = vectors[0]?.count || 0
		this.shape = [this.rows, this.columns]
		this.geometry = `${this.columns}x${this.rows}`
		this.isSquare = this.rows && (this.rows === this.columns)
	}


	// generics

	map(f) {
		return new Matrix(...this.vectors.map(f))
	}


	mapElements(f) {
		return Matrix.fromArray(
			this.vectors.map((vector, i) =>
				vector.elements.map((element, j) => f(element, i, j))
			)
		)
	}


	filter(f) {
		return new Matrix(...this.vectors.filter(f))
	}

	reduce(f) {
		return this.vectors.reduce(f)
	}	

	unary(f) {
		return this.map(vector => f(vector))
	}


	binary(matrix, f) {
		return this.map((vector, index) => f(vector, matrix.getRow(index)))
	}


	// accessors

	copy() {
		return this.map(vector => vector.copy())
	}

	get(row, column) {
		return this.vectors[row].get(column)
	}

	getRow(row) {
		return this.vectors[row]
	}

	set(row, column, value) {
		return this.vectors[row].set(column, value)
	}


	// matrix negation

	negate() {
		return this.unary(Operations.negate)
	}


	// matrix addition

	add(matrix) {
		return this.binary(matrix, Operations.add)
	}


	// matrix subtraction

	subtract(matrix) {
		return this.binary(matrix, Operations.subtract)
	}


	// matrix multiplication

	times(matrix) {
		return this.binary(matrix, Operations.times)
	}


	// matrix division

	divide(matrix) {
		return this.binary(matrix, Operations.divide)
	}


	// matrix dot product

	dot(matrix) {
		return Matrix.fromArray(MatrixProd(this.l, matrix.l))
	}


	// matrix exponentiation (point-wise)

	pow(exponent) {
		return this.unary(x => x.pow(exponent))
	}


	// matrix exponentiation 

	powd(exponent) {
		let result = this.copy()

		if (exponent === 0) {
			return Matrix.identity(this.rows)
		}

		for (let i=0; i < exponent - 1; i++) {
			result = result.dot(this)
		}

		return result
	}


	// matrix square 

	get square() {
		return this.pow(2)
	}

	get squared() {
		return this.powd(2)
	}

	// matrix equality

	equals(matrix) {
		return this.filter((vector, i) => !vector.equals(matrix.getRow(i))).rows === 0
	}


	// scalar addition

	increase(scalar) {
		return this.unary(vector => vector.increase(scalar))
	}


	// scalar subtraction

	decrease(scalar) {
		return this.unary(vector => vector.decrease(scalar))
	}


	// scalar multiplication

	scale(scalar) {
		return this.unary(vector => vector.scale(scalar))
	}


	// scalar division

	shrink(scalar) {
		return this.unary(vector => vector.shrink(scalar))
	}


	// vector multiplication
	
	transform(vector) {
		return this.dot(vector.transpose)
	}

	// vector addition/subtraction

	translate(vector) {
		return this.t.unary(row => row.add(vector)).t
	}

	// stringify

	toString() {
		return `(${this.vectors.join('\n ')})`
	}

	toArray() {
		return this.vectors.map(vector => vector.elements)
	}

	getColumn(i) {
		return new Vector(...this.vectors.map(vector => vector.get(i)))
	}

	toVector() {
		return this.getColumn(0)
	}

	get s() {
		return this.toString()
	}

	get l() {
		return this.toArray()
	}

	get v() {
		return this.toVector()
	}

	get transpose() {
		return Matrix.fill(this.columns, this.rows, 0).mapElements((_, i, j) => this.get(j, i))
	}

	get t() {
		return this.transpose
	}

	// static functions

	static fromArray(array) {
		const vectors = array.map(elements => new Vector(...elements))

		return new Matrix(...vectors)
	}


	static fill(rows, columns, value) {
		return Matrix.fromArray(
			Array(rows).fill(0).map(() => Array(columns).fill(value))
		)
	}

	static zero(n) {
		return Matrix.fill(n, n, 0)
	}

	static one(n) {
		return Matrix.fill(n, n, 1)
	}

	static identity(n) {
		const z = Matrix.zero(n)

		for (let i=0; i < n; i++) {
			z.set(i, i, 1)
		}

		return z
	}


	// determinants

	get det2d() {
		const a = this.get(0, 0)
		const b = this.get(0, 1)
		const c = this.get(1, 0)
		const d = this.get(1, 1)

		return a*d - b*c
	}


	get det3d() {
		const a = this.get(0, 0)
		const b = this.get(0, 1)
		const c = this.get(0, 2)

		const d = this.get(1, 0)
		const e = this.get(1, 1)
		const f = this.get(1, 2)

		const g = this.get(2, 0)
		const h = this.get(2, 1)
		const i = this.get(2, 2)

		return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
	}

	// inverses

	get inv2d() {
		const a = this.get(0, 0)
		const b = this.get(0, 1)
		const c = this.get(1, 0)
		const d = this.get(1, 1)

		const m = new Matrix(
			V(d, -b), 
			V(-c, a)
		)

		return m.shrink(m.det2d)
	}

	// 3x3 inverse: calculate the adjoint, then divide by the determinant.

	get inv3d() {

		const a = this.get(0, 0)
		const b = this.get(0, 1)
		const c = this.get(0, 2)
		const d = this.get(1, 0)
		const e = this.get(1, 1)
		const f = this.get(1, 2)
		const g = this.get(2, 0)
		const h = this.get(2, 1)
		const i = this.get(2, 2)

		// minors

		const m1 = new Matrix(V(e, f), V(h, i))
		const m2 = new Matrix(V(d, f), V(g, i))
		const m3 = new Matrix(V(d, e), V(g, h))
		const m4 = new Matrix(V(b, c), V(h, i))
		const m5 = new Matrix(V(a, c), V(g, i))
		const m6 = new Matrix(V(a, b), V(g, h))
		const m7 = new Matrix(V(b, c), V(e, f))
		const m8 = new Matrix(V(a, c), V(d, f))
		const m9 = new Matrix(V(a, b), V(d, e))

		// cofactors

		const m1d = + m1.det2d
		const m2d = - m2.det2d
		const m3d = + m3.det2d
		const m4d = - m4.det2d
		const m5d = + m5.det2d
		const m6d = - m6.det2d
		const m7d = + m7.det2d
		const m8d = - m8.det2d
		const m9d = + m9.det2d

		// adjoint

		const cofactors = new Matrix(
			V(m1d, m2d, m3d),
			V(m4d, m5d, m6d),
			V(m7d, m8d, m9d),
		)

		console.log('[+] COFACTOR MATRIX:')

		cofactors.tab

		console.log('[+] ADJOINT MATRIX (transpose of cofactors):')

		const adjoint = cofactors.t
		adjoint.tab

		console.log('[+] DETERMINANT: ')

		const determinant = this.det3d
		console.log('det(A) = ' + determinant)
		console.log('invdet(A) = ' + 1/determinant)

		console.log('[+] INVERSE:')

		const inverse = adjoint.shrink(determinant)
		inverse.tab

		return inverse
		
	}

	get inv() {
		return this.rows === 3
			? this.inv3d 
			: this.inv2d
	}


	// gramian matrix

	get gramian() {
		return this.dot(this.transpose)
	}


	// left and right pseudo-inverses

	get linv() {
		return this.gramian.inv.dot(this.t)
	}

	get rinv() {
		return this.t.dot(this.gramian.inv)
	}

	// misc

	get tab() {
		console.table(this.l)
	}

	get negative() {
		return this.negate()
	}


	// row operations 

	select(row, newRow) {
		const zeros = this.add(this.negative)
	
		newRow ?? (newRow = row)
		zeros.vectors[newRow] = this.getRow(row).copy()

		return zeros
	}

	ropAdd(sink, source, coeff=1) {
		return this.add( this.select(source, sink).scale(coeff) )
	}

	ropSwap(sink, source) {
		const copy = this.copy()

		copy.vectors[sink] = this.vectors[source].copy()
		copy.vectors[source] = this.vectors[sink].copy()

		return copy
	}

	ropScale(sink, scalar) {
		const copy = this.copy()

		copy.vectors[sink] = copy.vectors[sink].scale(scalar)
		return copy
	}

	// change of basis A --> (C^-1) . A . C
/*
	changeBasis(...basisVectors) {
		const changeOfBasisMatrix = new Matrix(...basisVectors).t
		

		return changeOfBasisMatrix.inv.dot(this).dot(changeOfBasisMatrix)
		
	}
*/

	// eigenstuffs (only for 2d matricies)

	get eigenvalues2d() {
		// solve using quadratic formula
		// general form is x^2 - (D + A)x + (AD - BC) = 0

		log(`
			  Av = kv 
 			  <===> Av = kIv 
			  <===> Av - kIv = 0
			  <===> (A - kI)v = 0
			  <===> Require v != 0, then det(A - kI) = 0
`)

		log(`General form: x^2 - (D + A)x + (AD - BC) = 0`)

		const A = this.get(0, 0)
		const B = this.get(0, 1)
		const C = this.get(1, 0)
		const D = this.get(1, 1)

		const a = 1
		const two_a = 2
		const b = -(D + A)
		const c = A*D - B*C

		const sqrt_b2_4ac = Math.sqrt(b ** 2 - 4*a*c)
		const x1 = (-b + sqrt_b2_4ac) / two_a
		const x2 = (-b - sqrt_b2_4ac) / two_a

		return [x1, x2]
	}

	static checkEigendecomposition(eigenvectorMatrix, eigenvalueMatrix) {
		return eigenvectorMatrix.dot(eigenvalueMatrix).dot(eigenvectorMatrix.inv).tab
	}


	// singular value decomposition

	static svd2d(A) {
		// A @ A.t

		console.log('[+] compute A @ A.t to get a square matrix')

		const AAT = A.t.dot(A)

		AAT.tab

		console.log('[+] compute eigendecomposition of (A.t @ A)')
		console.log('[+] (A @ A.t) = Q @ L @ (Q^-1)')
		console.log('[+] it can be shown that Q = V and L = Sigma')

		console.log('[*] attempting to find eigenvalues of (A.t @ A)...')
		
		const eigenvalues = AAT.eigenvalues2d

		if (eigenvalues) {
			console.log(`[+] found: ${eigenvalues}`)
		} else {
			log('unable to find eigenvalues')
		}

		console.log('[+] now go and find the eigenvectors yourself!')
		console.log('[+] after you have the eigenvectors, write the eigendecomposition of (A.t @ A):')
		console.log('[+] (A.t @ A) = Q @ L @ (Q^-1)')
		console.log('[+] for the SVD of A, use Sigma=L and V=Q')
		log('[+] finally write, A = U @ Sigma @ V.t')
	}
}


// Cramer's rule for SLE

class SLE {

	static cramer2d(A, b) {

		const Adet = A.det2d
		const A1 = A.copy()
		const A2 = A.copy()

		console.log('(0) det(A) = ' + Adet)
		console.log('(1) find x1...')
		console.log(`(1.1) insert ${b} into column 0 of A`)

		A1.set(0, 0, b.get(0))
		A1.set(1, 0, b.get(1))

		console.log('resulting matrix A1:')

		A1.tab

		const x1 = A1.det2d / Adet

		console.log('(1.2) x1 = det(A1)/det(A) = ' + x1)

		console.log('(2) find x2...')		
		console.log(`(2.1) insert ${b} into column 1 of A`)

		A2.set(0, 1, b.get(0))
		A2.set(1, 1, b.get(1))

		console.log('resulting matrix A2:')
		A2.tab

		const x2 = A2.det2d / Adet
		console.log('(2.2) x2 = det(A2)/det(A) = ' + x2)

		console.log('-------------------------------------------------------------')
		console.log(`(3) SOLUTION: x1 = ${x1}, x2 = ${x2}`)
	}


	static cramer3d(A, b) {
		const Adet = A.det3d
		const A1 = A.copy()
		const A2 = A.copy()
		const A3 = A.copy()

		// insert column vectors

		A1.set(0, 0, b.get(0))
		A1.set(1, 0, b.get(1))
		A1.set(2, 0, b.get(2))

		A2.set(0, 1, b.get(0))
		A2.set(1, 1, b.get(1))
		A2.set(2, 1, b.get(2))

		A3.set(0, 2, b.get(0))
		A3.set(1, 2, b.get(1))
		A3.set(2, 2, b.get(2))

		const x1 = A1.det3d / Adet
		const x2 = A2.det3d / Adet
		const x3 = A3.det3d / Adet

		console.log('x1 = ' + x1)
		console.log('x2 = ' + x2)
		console.log('x3 = ' + x3)

		return {
			Adet,
			A1,
			A2,
			A3,
			x1,
			x2,
			x3,
		}
	}
}


// eigendecomposition

// aliases

const ID2 = Matrix.identity(2)
const ID3 = Matrix.identity(3)
const ID4 = Matrix.identity(4)
const ID = Matrix.identity
const M = (...vectors) => new Matrix(...vectors)
const mm = Matrix
const gram = (...vectors) => M(...vectors).gramian





// general algebra

function findLine(x1, y1, x2, y2) {

	const rise = y2 - y1
	const run = x2 - x1
	const m = rise / run

	console.log('rise: ', rise)
	console.log('run: ', run)
	console.log('m: ', m)

	const c = y1 - m*x1

	console.log('c: ', c)

	return `y = ${m}x + ${c}`
}


function findLinePolar(m, c) {

	console.log(`rsin(theta) = (m * r)cos(theta) + c`)
	
	console.log(`rsin(theta) - (m * r)cos(theta) = c`)
	console.log(`r(sin(theta) - mcos(theta)) = c`)
	console.log(`r = c / (sin(theta) - m*cos(theta))`)

	return `r = ${c} / (sin(theta) - ${m}cos(theta))`										
}


function fromPolar(r, radians) {

	const x = r * Math.cos(radians)
	const y = r * Math.sin(radians)

	console.log('x = r * cos(theta): ' + x)
	console.log('y = r * sin(theta): ' + y)

	return [x, y]
}


function toPolar(x, y) {
	const r = Math.sqrt(x ** 2 + y ** 2)
	let radians

	console.log('r = hypot(x, y) = sqrt(x^2 + y^2): ' + r)
	
	if (y >= 0 && r !== 0) {
		console.log('y >= 0 and r != 0')

		radians = Math.acos(x / r)
		console.log('therefore, theta = arccos(x / r): ' + radians)

	} else if (y < 0 && r !== 0){

		console.log('y < 0 and r != 0')
		radians = -Math.acos(x / r)

		console.log('therefore, theta = -arccos(x / r): ' + radians)
	} else if (r == 0) {
		console.log('r is 0')

		radians = null
		console.log('therefore, theta = undefined')
	}

	return [r, radians]
}


function toBary([px, py], [ax, ay], [bx, by], [cx, cy]) {

	const l0 = ((by-cy)*(px-cx) + (cx-bx)*(py-cy)) / ((by-cy)*(ax-cx)+(cx-bx)*(ay-cy))
	const l1 = ((cy-ay)*(px-cx)+(ax-cx)*(py-cy)) / ((by-cy)*(ax-cx)+(cx-bx)*(ay-cy))
	const l2 = 1 - l0 - l1

	console.log('lambda 1: (y2 - y3)(x - x3) + (x3 - x2)(y - y3) / det(T) = ' + l0)
	console.log('lambda 2: (y3 - y1)(x - y3) + (x1 - x3)(y - y3) / det(T) = ' + l1)
	console.log('lambda 3: 1 - (lambda 1) - (lambda 2) = ' + l2)

	return [l0, l1, l2]
}


function fromBary([lambda1, lambda2, lambda3], [vx1, vy1], [vx2, vy2], [vx3, vy3]) {

	const x = (lambda1 * vx1) + (lambda2 * vx2) + (lambda3 * vx3)
	const y = (lambda1 * vy1) + (lambda2 * vy2) + (lambda3 * vy3)

	console.log('x = (lambda 1)x1 + (lambda 2)x2 + (lambda 3)x3 = ' + x)
	console.log('y = (lambda 1)y1 + (lambda 2)y2 + (lambda 3)y3 = ' + y)

	return [x, y]
}

function helpDistanceBetweenPointAndParametricLine() {
	console.log(`

		1. Let L(t) be a line, e.g. L(t) = (2t, 3t - 1, -t)
		2. Let P be a point in the same dimension as L, e.g. (1, 2, 3)

		3. To calculate the distance between L and P, treat both as vectors:
		   L(t) = V(2t, 3t - 1, -t),  P = V(1, 2, 3)
	
		4. Then subtract them, D = L - P.
		5. Calculate the magnitude-squared of D, i.e. d^2 = ||D||^2
		6. Now find t which minimises d^2.
		7. Once found, plug t back into d^2 to get the final value for d^2 and hence d.
	`)
}


function classifyCriticalPoint(x, y, fxx, fyy, fxy) {

	const check1 = fxx(x,y)*fyy(x,y) - fxy(x,y)*fxy(x,y)
	const check2 = fxx(x,y) + fyy(x,y)

	if (check1 < 0) {
		console.log('(fxx * fyy) - (fxy * fxy) < 0')
		console.log('therefore, it\'s a SADDLE point')

		if (check2 < 0) {
			console.log('fxx + fyy < 0')
			console.log('therefore, it\'s a RIDGE SADDLE')
		}

		else if (check2 == 0) {
			console.log('fxx + fyy == 0')
			console.log('therefore, it\'s a BALANCED SADDLE')
		}

		else if (check2 > 0) {
			console.log('fxx + fyy < 0')
			console.log('therefore, it\'s a TROUGH SADDLE')
		}
	}

	else if (check1 == 0) {
		console.log('(fxx * fyy) - (fxy * fxy) == 0')
		console.log('therefore, it\'s a SHOE SURFACE')
	}

	else if (check1 > 0) {
		console.log('(fxx * fyy) - (fxy * fxy) > 0')
		console.log('therefore, it\'s an EXTREMUM')


		if (check2 < 0) {
			console.log('fxx + fyy < 0')
			console.log('therefore, it\'s a MAXIMUM')
		}

		else if (check2 > 0) {
			console.log('fxx + fyy < 0')
			console.log('therefore, it\'s a MINIMUM')
		}
	}
}

function polarToRectangularForm() {
		console.log(`

Replace r^2 with (x^2 + y^2)
Replace rcos(theta) with x
Replace rsin(theta) with y
`)
}

// y - y1 = m(x - x1)
// y = m(x - x1) + y1
// y = mx - x1 + y1
// c = -x1 + y1

function findLine2(m, x1, y1) {

log(`
// y - y1 = m(x - x1)
// y = m(x - x1) + y1
// y = mx - x1 + y1
// c = -x1 + y1
`)

	return `y = ${m}x + ${-x1 + y1}`
}

function tg(m) {
	return -1/m
}

function proj(u, v) {
	if (u && v) return u.scale(u.dot(v) / u.dot(u))
}


function orthonormalize(v1, v2, v3, v4) {

	const u1 = v1
	const u2 = v2?.subtract(proj(u1, v2))
	const u3 = v3?.subtract(proj(u1, v3) + proj(u2, v3))
	const u4 = v4?.subtract(proj(u1, v4) + proj(u2, v4))

	log('Gram–Schmidt process:')

	log('proj(u, v) = u * (u . v)/(u, u)')
	log('u1 = v1')
	log('u2 = v2 - proj(u1, v2)')
	log('u3 = v3 - proj(u1, v3) - proj(u2, v3)')
	log('u4 = v4 - proj(u1, v4) - proj(u2, v4)')

	log('{ ei } = { ui.normal }')

	return {
		u1,
		u2,
		u3,
		u4,
	}
}

// statistics section 

window.logging = true

window.togglelog = () => window.logging = !window.logging

function log(m) {
	if (window.logging) console.log(m)
}



const Parsers = {

	SeparatedValues(string, separator=/\s+/) {
		const lines = string.trim().split('\n').filter(line => line.trim())
		const vectors = []

		lines.forEach(line => {
			const elements = line.trim().split(separator).map(element => parseFloat(element))
			const vector = new Vector(...elements)

			vectors.push(vector)
		})

		return vectors
	}
}




// helpers...


function vectorize(object) {
	if (object instanceof Vector) return object
	if (object instanceof Object) return new Vector(...object)
	
	return new Vector(object)
}


// inter-quartile range

function IQR(vector) {

	const Q3 = vector.percentile(75)
	const Q1 = vector.percentile(25)

	return Q3 - Q1	
}


// sample coefficient of variation - the ratio of the standard deviation to the mean.

function CV(vector) {
	return Math.sqrt(vector.svariance) / vector.mean
}


// create a vector from a frequency distribution

function vectorFromFrequency(keys, values) {

	const elements = []

	keys.forEach((key, i) => {
		let freq = values[i]
		elements.push(...Array(freq).fill(key))
	})

	return new Vector(...elements)
}


// aliases 

const v = vec = vector = vectorize
const V = (...a) => v(a)
const $ = dot = (...a) => (...b) => v(a).dot(v(b))
const mu = E = mean = (...a) => v(a).mean
const p2 = (...a) => v(a).square()
const VAR = Var = a => v(a).svariance
const VARP = a => v(a).variance()
const S = a => Math.sqrt(v(a).variance)
const vff = VFF = vectorFromFrequency
const CSV = a => Parsers.SeparatedValues(a, separator=/\s*,\s*/)
const SV = Parsers.SeparatedValues
const TSV = a => Parsers.SeparatedValues(a, separator=/\t+/)

function assert(value, message='assertion error') {
	if (value === false) throw message
}


class Event {

	constructor(...outcomes) {
		this.outcomes = new Set(outcomes)
	}

	get size() {
		return this.outcomes.size
	}

	map(f) {
		return new Event(...[...this.outcomes].map(f))
	}

	filter(predicate) {
		return new Event(...[...this.outcomes].filter(predicate))
	}

	reduce(f) {
		return [...this.outcomes].reduce(f, 0)
	}

	forEach(f) {
		return [...this.outcomes].forEach(f)
	}

	has(outcome) {
		return this.outcomes.has(outcome)
	}

	union(event) {
		return new Event(...[...this.outcomes, ...event.outcomes])
	}

	intersection(event) {
		return this.filter(event.has.bind(event))
	}

	difference(event) {
		return this.filter(outcome => !event.has(outcome))
	}

	product(event, outcomeCombiner=StringOutcomeCombiner) {
		const productOutcomes = []

		this.forEach(x => event.forEach(y => productOutcomes.push(outcomeCombiner.combine(x, y))))
		return new Event(...productOutcomes)
	}

	isSubset(event) {
		return this.difference(event).size === 0
	}
}


class RandomVariable {

	constructor(sampleSpace, transformation) {

		this.sampleSpace = sampleSpace
		this.transformation = transformation
	}

	get support() {
		return this.sampleSpace.map(this.transformation).outcomes
	}

	when(predicate) {
		return this.sampleSpace.filter(outcome => predicate(this.transformation(outcome)))
	}

	lt(value) {
		return this.when(x => x < value)
	}

	gt(value) {
		return this.when(x => x > value)
	}

	eq(value) {
		return this.when(x => x === value)
	}

	ne(value) {
		return this.when(x => x !== value)
	}
}

function expected(transformation, probabilitySpace) {
	const distribution = pdist(transformation, probabilitySpace)

	return distribution.map((value, i) => (i + 1) * value)
		.reduce((a, b) => a + b, 0)
}

class ProbabilityMeasure {

	P(outcome) {

		throw 'mapping not defined'
	}
}


class FairProbabilityMeasure extends ProbabilityMeasure {

	constructor(numberOfOutcomes) {
		super()

		this.numberOfOutcomes = numberOfOutcomes
	}

	p(outcome) {
		return 1 / this.numberOfOutcomes
	}
}


class MapProbabilityMeasure extends ProbabilityMeasure {

	constructor(mapping) {
		super()

		this.mapping = mapping
	}

	p(outcome) {
		return this.mapping[outcome]
	}
}


class ProductProbabilityMeasure extends ProbabilityMeasure {


	constructor(leftProbabilityMeasure, rightProbabilityMeasure, outcomeCombiner) {
		super()

		this.p = outcome => leftProbabilityMeasure.p(outcomeCombiner.left(outcome)) * rightProbabilityMeasure.p(outcomeCombiner.right(outcome))
		
	}
}


class ProbabilitySpace {

	constructor(sampleSpace, probabilityMeasure) {

		this.sampleSpace = sampleSpace
		this.probabilityMeasure = probabilityMeasure ?? new FairProbabilityMeasure(sampleSpace.size)
	}

	p(event) {
		return event.reduce((sum, x) => sum + this.probabilityMeasure.p(x))
	}

	pp(predicate) {
		return this.p(this.sampleSpace.filter(predicate))
	}


	product(other, outcomeCombiner=StringOutcomeCombiner) {

		return new ProbabilitySpace(
			sampleSpace=this.sampleSpace.product(other.sampleSpace, outcomeCombiner),
			probabilityMeasure=new ProductProbabilityMeasure(this.probabilityMeasure, other.probabilityMeasure, outcomeCombiner)
		)
	}

	variable(transformation) {
		return new RandomVariable(this.sampleSpace, transformation)
	}
}


class OutcomeCombiner {

	static combine(x, y) {
		throw 'not implemented'
	}
}


class StringOutcomeCombiner extends OutcomeCombiner {

	static combine(x, y) {
		return `${x},${y}`
	}

	static left(string) {
		return string.split(',')[0]
	}

	static right(string) {
		return string.split(',')[1]
	}
}


class VectorOutcomeCombiner extends OutcomeCombiner {

	static combine(x, y) {
		return new Vector(x, y)
	}

	static left(vector) {
		return vector.get(0)
	}

	static right(vector) {
		return vector.get(1)
	}
}

// helpers


// compute the probability distribution of a random variable

function pdist(transformation, probabilitySpace) {

	const variable = probabilitySpace.variable(transformation)

	return [...variable.support]
		.map(value => variable.eq(value))
		.map(probabilitySpace.p.bind(probabilitySpace))
}


// an alias to make it clear when we're talking about the root event/sample space.

const RootEvent = SampleSpace = Event

// more aliases

const RE = (...a) => new ProbabilitySpace(...a)
const Ev = (...a) => new Event(...a)
const e = Math.E
const PI = Math.PI


const dice = n => new ProbabilitySpace(
	sampleSpace=new Event(...Array(n).fill(0).map((_, i) => i + 1)),
	probabilityMeasure=new FairProbabilityMeasure(n)
)



// conditional probability

ProbabilitySpace.prototype.pcond = function(predicate1, predicate2) {

	return this.pp(outcome => predicate1(outcome) && predicate2(outcome)) / this.pp(predicate2)
}

// useful probability space

// two dice rolls

const TwoDice = dice(6).product(dice(6), VectorOutcomeCombiner)

TwoDice.sum = TwoDice.variable(x => x.get(0) + x.get(1))
TwoDice.product = TwoDice.variable(x => x.get(0) * x.get(1))
TwoDice.match = TwoDice.variable(x => x.get(0) == x.get(1))
TwoDice.maximum = TwoDice.variable(x => Math.max(x.get(0), x.get(1)))


// permutations and combinatorics

function fact(n) {
	return n ? Array(n).fill(0).map((_, i) => i + 1).reduce(Operations.times) : 1
}


// nPr - the number of ways you can arrage r distinct objects from a set of n objects.

function npr(n, r) {
	return fact(n) / fact(n - r)
}


// nCr - the number of ways you can *choose* r distinct objects from a set of n object. Two permutations with the 
// same elements but different ordering are considered the same *choice*.

// each choice is a set of r unique elements. Hence, there are r! ways to arrange each set. Hence, in total, the 
// total number of ways to arrange r objects from a set of n objects is nCr * r!. Then nCr = nPr / r!
// 
//

function ncr(n, r) {
	return npr(n, r) / fact(r)
}


// bayes theorem for disease tests

function infected_given_result(prior_infected, sensitivity, specificity, classes=['infected', 'clean']) {


	const infected_class = classes[0]
	const clean_class = classes[1]


	// priors

	const p_infected = prior_infected
	const p_clean = 1 - prior_infected
	
	log('priors:')
	log(`p(${infected_class}) = ${prior_infected}`)
	log(`p(${clean_class}) = 1 - p(${infected_class}) = ${p_clean}`)

	// likelihoods 

	const p_positive_given_infected = sensitivity
	const p_negative_given_infected = 1 - sensitivity
	const p_positive_given_clean = 1 - specificity
	const p_negative_given_clean = specificity

	log('likelihoods:')
	log(`p(positive | ${infected_class}) = sensitivity = ${p_positive_given_infected}`)
	log(`p(negative | ${infected_class}) = 1 - sensitivity = ${p_positive_given_infected}`)
	log(`p(positive | ${clean_class}) = 1 - specificity = ${p_positive_given_clean}`)
	log(`p(negative | ${clean_class}) = specificity = ${p_negative_given_clean}`)

	// numerators

	const n_positive_given_infected = p_infected * p_positive_given_infected
	const n_negative_given_infected = p_infected * p_negative_given_infected
	const n_positive_given_clean = p_clean * p_positive_given_clean
	const n_negative_given_clean = p_clean * p_negative_given_clean

	// marginals 

	const m_positive = n_positive_given_infected + n_positive_given_clean
	const m_negative = n_negative_given_infected + n_negative_given_clean

	log('marginals:')
	log(`p(positive) = p(${infected_class})p(positive | ${infected_class}) + p(${clean_class})p(positive | ${clean_class}) = ${m_positive}`)
	log(`p(negative) = p(${infected_class})p(negative | ${infected_class}) + p(${clean_class})p(negative | ${clean_class}) = ${m_negative}`)

	// posteriors

	const p_infected_given_positive = n_positive_given_infected / m_positive
	const p_infected_given_negative = n_negative_given_infected / m_negative
	const p_clean_given_positive = n_positive_given_clean / m_positive
	const p_clean_given_negative = n_negative_given_clean / m_negative

	log(`posteriors:`)
	log(`p(${infected_class} | positive) = p(${infected_class})p(positive | ${infected_class}) / p(positive) = ${p_infected_given_positive}`)
	log(`p(${infected_class} | negative) = p(${infected_class})p(negative | ${infected_class}) / p(negative) = ${p_infected_given_negative}`)
	log(`p(${clean_class} | positive) = p(infected)p(positive | ${clean_class}) / p(positive) = ${p_clean_given_positive}`)
	log(`p(${clean_class} | negative) = p(infected)p(positive | ${clean_class}) / p(positive) = ${p_clean_given_negative}`)

	return {
		[`p(${infected_class} | positive)`]: p_infected_given_positive,
		[`p(${infected_class} | negative)`]: p_infected_given_negative,
		[`p(${clean_class} | positive)`]: p_clean_given_positive,
		[`p(${clean_class} | negative)`]: p_clean_given_negative,
	}
}


// various discrete parametric distributions...

class ProbabilityDistribution {

	constructor(parameters) {
		this.parameters = parameters
	}

	p(x) {
		throw 'not implemented'
	}
	
	cdf(x) {
		throw 'not implemented'
	}

	pp(predicate) {
		return this.support
			.filter(predicate)
			.map(value => this.p(value))
			.reduce((a, b) => a + b, 0)	
	}

	prange(start, stop) {
		return Array(stop - start)
			.fill(0)
			.map((_, i) => this.p(start + i))
			.reduce((a, b) => a + b, 0)
	}

	get pdist() {
		return this.support.map(value => this.p(value))
	}


	get mean() {
		return this.support
			.map(value => value * this.p(value))
			.reduce((a, b) => a + b, 0)
	}

	get mode() {
		return new Vector(...this.pdist).argmax
	}

	get median() {
		return 'not implemented'
	}
}


class BernoulliDistribution extends ProbabilityDistribution {


	p(x) {
		return x === 1 ? this.parameters.p : (1 - this.parameters.p)
	}

	get support() {
		return [0, 1]
	}

	get mean() {
		log('Mean of Bernoulli: u = p')

		return this.parameters.p
	}
	
	get mode() {
		log('Mode of Bernoulli:')
		log('0 if p < 0.5')
		log('1 if p > 0.5')
		log('0 and 1 if p == 0.5')

		if (p == 0.5) throw 'bimodal at 0 and 1'

		return this.parameters.p < 0.5 ? 0 : 1 // this is technically wrong for p=0.5 where the distribution is bimodal.
	}

	get variance() {
		return this.parameters.p * (1 - this.parameters.p)
	}
}


class BinomialDistribution extends ProbabilityDistribution {

	p(x) {
		return ncr(this.parameters.n, x) * (this.parameters.p ** x) * ((1 - this.parameters.p) ** (this.parameters.n - x))
	}

	cdf(x) {
		return this.prange(0, x)
	}

	get support() {
		return Array(this.parameters.n + 1)
			.fill(0)
			.map((_, i) => i)
	}

	get mean() {
		log('Mean of Binomial: u = np')

		return this.parameters.n * this.parameters.p
	}

	get mode() {

		log('Mode of Binomial: floor((n + 1) * p)')

 		return Math.floor((this.parameters.n + 1) * this.parameters.p)
	}

	get median() {
		return Math.floor(this.mean)
	}

	get variance() {
		log('Variance of Binomial: np(1 - p)')

		return this.parameters.n * this.parameters.p * (1 - this.parameters.p)
	}

	get normal() {
		const mu = this.parameters.n * this.parameters.p
		const sigma2 = this.parameters.n * this.parameters.p * (1 - this.parameters.p)

		log('Approxiamte as a normal distribution:')
		log('mu = np = ' + mu)
		log('sigma^2 = npq = ' + sigma2)
		log('Note: remember to use the interpolated values when doing P(X < x)')

		return new NormalDistribution({mu, sigma: Math.sqrt(sigma2)})
	}
}


class GeometricDistribution extends ProbabilityDistribution {

	p(x) {
		log('PDF of Geometric: P(x = x) = p * (1 - p)^(x - 1)	(drop the -1 if not inclusive)')

		return this.parameters.p * ((1 - this.parameters.p) ** (x - this.parameters.inclusive)) 
	}

	cdf(x) {
		log('CDF of Geometric: P(X < x) = 1 - (1 - p)^(x + 1)    (drop the 1 if not inclusive)')

		return 1 - (1 - this.parameters.p) ** (x + this.parameters.inclusive)
	}

	get mean() {
		log('Mean of Geometric: u = 1/p. If not inclusive then: u = (1 - p)/p')

		return this.parameters.inclusive
			? (1 / this.parameters.p)
			: ((1 - this.parameters.p) / this.parameters.p)
	}

	get variance() {
		log('Variance of Geometric: (1 - p) / p^2')

		return (1 - this.parameters.p) / (this.parameters.p ** 2)
	}

	get mode() {
		log('Mode of Geometric: 1 if inclusive, 0 otherwise')

		return this.inclusive ? 1 : 0
	}

	get support() {
		throw 'distribution has infinite support'
	}
}


class PoissonDistribution extends ProbabilityDistribution {


	p(x) {
		log('PDF of Poisson: P(x = x) = (lambda^x * e^-lambda) / x!')

		return Math.pow(this.parameters.lambda, x) * Math.exp(-this.parameters.lambda) / fact(x)
	}

	cdf(x) {
		return this.prange(0, x)
	}

	get mean() {
		log('Mean of Poisson: u = lambda')

		return this.parameters.lambda
	}

	get mode() {
		log('Mode of Poisson: floor(lambda)')

		return Math.floor(this.parameters.lambda)
	}

	get median() {
		log('Median of Poisson: approx. floor(lambda + 1/3 - (0.02 / lambda))')

		return Math.floor((this.parameters.lambda + 1/3) - (0.02 / this.parameters.lambda))
	}

	get variance() {
		log('Variance: lambda')

		return this.parameters.lambda
	}

	get exponential() {
		log('Convert to exponential: X ~ Exp(lambda)')

		return new ExponentialDistribution({ lambda: this.parameters.lambda })
	}
}



// convolutions of distributions (sums of random variables)

class Convolutions {

	// sum of n independent binomial variables with the same value for `p`

	static binomial(...variables) {
		const total_n = variables.map(x => x.parameters.n).reduce((a, b) => a + b, 0)
		const p = variables[0].parameters.p


		log('Sum of n independent binomial swith the same value for p:')
		log('n\' = sum of individual ns')
		log('p\' = p')
		log('X ~ B(n\', p\')')
 
		return new BinomialDistribution({
			n: total_n,
			p
		})		
	}


	// sum of n independent bernoulli variables

	static bernoulli(...variables) {
		log('Sum of n independent Bernoulli variables, each with the same value for p:')
		log('X ~ B(# of variables, p)')

		return new BinomialDistribution({
			n: variables.length, 
			p: variables[0].parameters.p
		})
	}


	// sum of n independent poisson variables

	static poisson(...variables) {
		log('Sum of n independent Poisson variables:')
		log('X ~ Poi(lambda = sum of individual lambdas)')

		return new PoissonDistribution({
			lambda: variables.map(variable => variable.parameters.lambda)
				.reduce((a, b) => a + b, 0)
		})
	}


	// sum of normals

	static normal(...variables) {
		const sum_of_variances = variables.map(variable => variable.parameters.sigma ** 2)
			.reduce((a, b) => a + b, 0)

		const sum_of_means = variables.map(variable => variable.parameters.mu)
			.reduce((a, b) => a + b, 0)

		log('Sum of n normals:')
		log('u = sum of individual means')
		log('sigma^2 = sum of variances')
		log(`X ~ N(mu = ${sum_of_means}, sigma^2 = ${sum_of_variances})`) 

		return new NormalDistribution({
			mu: sum_of_means,
			sigma: Math.sqrt(sum_of_variances)
		})
	}
}

// aliases for distributions

const Ber = p => new BernoulliDistribution({ p })
const Geo = (p, inclusive) => new GeometricDistribution({ p, inclusive: inclusive ?? 1 })
const B = (n, p) => new BinomialDistribution({ n, p })
const Poi = lambda => new PoissonDistribution({ lambda })


// continuous distributions

class ContinuousProbabilityDistribution {

	constructor(parameters) {
		this.parameters = parameters
	}

	pdf(x) {
		return 'not implemented'
	}

	cdf(x) {
		return 'not implemented'
	}

	q(p) {
		throw 'not implemented'
	}

	mean() {
		return 'not implemented'
	}

	mode() {
		return 'not implemented'
	}

	variance() {
		return 'not implemented'
	}

	prange(start, stop) {
		return this.cdf(stop) - this.cdf(start)
	}
}


// the error function 

function erfc(x) {

	const z = Math.abs(x);
	const t = 1 / (1 + z / 2);

	const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 +
		t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
		t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
		t * (-0.82215223 + t * 0.17087277)))))))))

	return x >= 0 ? r : 2 - r
}




function ierfc(x){
        var z;
        var a  = 0.147;                                                   
        var the_sign_of_x;
        if(0==x) {
            the_sign_of_x = 0;
        } else if(x>0){
            the_sign_of_x = 1;
        } else {
            the_sign_of_x = -1;
        }

        if(0 != x) {
            var ln_1minus_x_sqrd = Math.log(1-x*x);
            var ln_1minusxx_by_a = ln_1minus_x_sqrd / a;
            var ln_1minusxx_by_2 = ln_1minus_x_sqrd / 2;
            var ln_etc_by2_plus2 = ln_1minusxx_by_2 + (2/(Math.PI * a));
            var first_sqrt = Math.sqrt((ln_etc_by2_plus2*ln_etc_by2_plus2)-ln_1minusxx_by_a);
            var second_sqrt = Math.sqrt(first_sqrt - ln_etc_by2_plus2);
            z = second_sqrt * the_sign_of_x;
        } else { // x is zero
            z = 0;
        }
  return z;
}

function probit(p) {
	return Math.sqrt(2) * ierfc(2*p - 1)
}



// the normal distribution

class NormalDistribution extends ContinuousProbabilityDistribution {

	pdf(x) {
		const m = this.parameters.sigma * Math.sqrt(2 * Math.PI)
		const e = Math.exp(-Math.pow(x - this.parameters.mu, 2) / (2 * (this.parameters.sigma ** 2)))

		return e / m 
	}

	cdf(x) {
		return 0.5 * erfc(-(x - this.parameters.mu) / (this.parameters.sigma * Math.sqrt(2)))
	}

	qwithin(p) {
		log('How many standard deviations do we need to me away from mu for p to be ' + p + '?')

		return probit((p + 1) / 2)
	}

	q(p) {
		log('Quantile function: Q(p) = mu + sigma * probit(p)')
		log(`= ${this.parameters.mu} + ${this.parameters.sigma} * ${probit(p)}`)

		return this.parameters.mu + this.parameters.sigma*probit(p)
	}

	get mean() {
		return this.parameters.mu
	}

	get mode() {
		return this.parameters.mu
	}

	get median() {
		return this.parameters.mu
	}

	get variance() {
		return this.parameters.sigma ** 2
	}

	standardise(x) {
		log('Convert to a standard normal deviate')
		log('P(X < x) = P( Z < (x - mu)/sigma ) = P(Z < z)')
		log(`z = ${x - (this.parameters.mu / this.parameters.sigma)}`)
	}
}


class ExponentialDistribution extends ContinuousProbabilityDistribution {

	pdf(x) {
		log('PDF of Exponential: P(x = x) = lambda * e^(-lambda * x)')

		return this.parameters.lambda * Math.exp(-x * this.parameters.lambda)
	}

	cdf(x) {
		log('CDF of Exponential: P(x < x) = 1 - e^(-lambda * x)')

		return x >= 0
			? (1 - Math.exp(-x * this.parameters.lambda))
			: 0
	}

	q(p) {
		return -Math.ln(1 - p) / this.parameters.lambda
	}

	get mean() {
		log('Mean of Exponential: u = 1 / lambda')

		return 1 / this.parameters.lambda
	}

	get mode() {
		return 0
	}

	get median() {
		log('Median of Exponential: ln(2) / lambda')

		return Math.LN2 / this.parameters.lambda
	}

	get variance() {
		log('Variance of Exponential: 1 / lambda^2')

		return 1 / (this.parameters.lambda ** 2)
	}	
}


class ErlangDistribution extends ContinuousProbabilityDistribution {

	pdf(x) {
		return this.parameters.lambda * Math.pow(x, this.parameters.n - 1) * Math.exp(-this.parameters.lambda * x) / fact(this.parameters.n - 1)
	}

	cdf(x) {
		return 
	}
}

class UniformContinuousDistribution extends ContinuousProbabilityDistribution {

	pdf(x) {
		log('PDF of uniform continous: 1 / (b - a) if x in [a, b] and 0 otherwise')

		return (x < this.parameters.start || x > this.parameters.stop) 
			? 0 
			: 1 / (this.parameters.stop - this.parameters.start)
	}

	cdf(x) {
		log('CDF of uniform continous: (x - a) / (b - a) if x in [a, b] and 0 otherwise')

		if (x < this.parameters.start) return 0
		if (x > this.parameters.stop) return 1
		
		return (x - this.parameters.start) / (this.parameters.stop - this.parameters.start)
	}

	get mean() {
		return 0.5 * (this.parameters.start + this.parameters.stop)
	}

	get median() {
		return this.mean
	}

	get mode() {
		throw 'distribution is not unimodal, specially mode is any value in (start, stop)'
	}

	get variance() {
		log('Variance of uniform continuous: 1/12 * (b - a)^2')

		return 1/12 * ((this.parameters.stop - this.parameters.start) ** 2)
	}
}


// student's t distribution

function LogGamma(Z) {
	with (Math) {
		var S=1+76.18009173/Z-86.50532033/(Z+1)+24.01409822/(Z+2)-1.231739516/(Z+3)+.00120858003/(Z+4)-.00000536382/(Z+5);
		var LG= (Z-.5)*log(Z+4.5)-(Z+4.5)+log(S*2.50662827465);
	}
	return LG
}

function Betinc(X,A,B) {
	var A0=0;
	var B0=1;
	var A1=1;
	var B1=1;
	var M9=0;
	var A2=0;
	var C9;
	while (Math.abs((A1-A2)/A1)>.00001) {
		A2=A1;
		C9=-(A+M9)*(A+B+M9)*X/(A+2*M9)/(A+2*M9+1);
		A0=A1+C9*A0;
		B0=B1+C9*B0;
		M9=M9+1;
		C9=M9*(B-M9)*X/(A+2*M9-1)/(A+2*M9);
		A1=A0+C9*A1;
		B1=B0+C9*B1;
		A0=A0/B1;
		B0=B0/B1;
		A1=A1/B1;
		B1=1;
	}
	return A1/A
}

function TCDF(X, df) {

	let A, S, Z, tcdf
	
    with (Math) {	

		if (df<=0) {
			alert("Degrees of freedom must be positive")
		} else {
			A=df/2;
			S=A+.5;
			Z=df/(df+X*X);
			BT=exp(LogGamma(S)-LogGamma(.5)-LogGamma(A)+A*log(Z)+.5*log(1-Z));
			if (Z<(A+1)/(S+2)) {
				betacdf=BT*Betinc(Z,A,.5)
			} else {
				betacdf=1-BT*Betinc(1-Z,.5,A)
			}
			if (X<0) {
				tcdf=betacdf/2
			} else {
				tcdf=1-betacdf/2
			}
		}
	}
    return tcdf
}


class StudentsTDistribution extends ContinuousProbabilityDistribution {


	cdf(x) {
		return TCDF(x, this.parameters.degreesOfFreedom || 1)
	}
}

function createContinuousProbabilityDistribution(pdf, cdf) {

	return class extends ContinuousProbabilityDistribution {

		pdf(x) {
			return pdf.bind(this)(x)
		}

		cdf(x) {
			return cdf.bind(this)(x)
		}
	}
}

function createContinuousProbabilityDistribution(support, p) {

	return class extends ContinuousProbabilityDistribution {

		pdf(x) {
			return pdf.bind(this)(x)
		}

		cdf(x) {
			return cdf.bind(this)(x)
		}
	}
}



function normalDifference(U, V) {
	log('Difference between two normals: U - V')
	log('mu = U.mu - V.mu')
	log('sigma^2 = U.sigma^2 + V.sigma^2')

	return new NormalDistribution({
		mu: U.parameters.mu - V.parameters.mu,
		sigma: Math.sqrt(U.parameters.sigma ** 2 + V.parameters.sigma ** 2)
	}) 
}


const N = (mu, sigma) => new NormalDistribution({ mu, sigma })
const Exp = lambda => new ExponentialDistribution({ lambda })
const U = (start, stop) => new UniformContinuousDistribution({ start, stop })
const CPD = createContinuousProbabilityDistribution
const T = degreesOfFreedom => new StudentsTDistribution({ degreesOfFreedom })

// central limit theorem

const CLT = function centralLimitTheorem(sampleSize, sampleMean, sampleStandardDeviation) {
	log('Central Limit Theorem:')
	log('mu = sample mean')
	log('sigma^2 = (sample standard deviation)^2 / sampleSize')
	
	return new NormalDistribution({ mu: sampleMean, sigma: sampleStandardDeviation / Math.sqrt(sampleSize) })
}


const tolorance = (mean, standardDeviation, tolorance) => {

	const X = N(mean, standardDeviation)
	const z = X.qwithin(tolorance)

	return `μ ± ${standardDeviation * z}σ`
}


const CI = {
	mean(vector, confidence=.95) {
		const sampleMean = vector.mean
		const sampleVariance = vector.svariance
		const sampleStdev = Math.sqrt(sampleVariance)
		const sampleSize = vector.count
		
		log('Variables:')
		log('* Sample mean (x_) = ' + sampleMean)
		log('* Sample standard deviation (s) = ' + sampleStdev)
		log('* Sample size (n) = ' + sampleSize)

		if (sampleSize < 30) {
			log('Assumptions:')
			log('* Data are independent')
			log('* Data are from an approximately normal distribution')

			log('Calculation:')
			log(`* We need to find a constant c such that there's a ${confidence} chance that the true mean lies within c distance of the sample mean.`)
			console.log('* Sample size too small for CLT (< 30).')
			console.log('* We need to use a T distribution instead.')

			console.log('|  See https://www.statskingdom.com/distribution-calculator.html')
			console.log(`|  distribution = T, df=${sampleSize - 1}, mode = p(X <= x1), p(X <= x2)`)
			console.log(`|  p1 = ${.5 - confidence/2}`)
			console.log(`|  p2 = ${.5 + confidence/2}`)

			const t = parseFloat(prompt('t value: '))
			const c = t * sampleStdev / Math.sqrt(sampleSize)

			log(`* P(t < T < t) = ${confidence}`)
			log('* We have that value t = ' + t)
			log(`* Finally, c = t * s/sqrt(n) = ${c}`)

			return [sampleMean - c, sampleMean + c]
			
		}

		log('Assumptions:')
		log('* Data are independent')
		log('* Data are numerous enough for the CLT to apply.')


		togglelog()
		const X = CLT(sampleSize, sampleMean, sampleStdev)
		const z = X.qwithin(confidence)
		const c = z * X.parameters.sigma
		togglelog()

		log('Calculation:')
		log(`* We need to find a constant c such that there's a ${confidence} chance that the true mean lies within c distance of the sample mean.`)
		log(`* Convert to a standard normal distribution: Z = (x_ - mu) / (s / sqrt(n)) ~ N(0, 1)`)
		log(`* P(z < Z < z) = ${confidence}`)
		log(`* Using the z-table, we have that z = ${z}`)
		log(`* Finally, c = z * s/sqrt(n) = ${c}`)

		return [sampleMean - c, sampleMean + c] 	
	}
}


function pValueTest(sampleSize, sampleMean, sampleStandardDeviation, hypothesisMean, significance=5/100, hypothesisType=0) {

	const testStatistic = (sampleMean - hypothesisMean) / (sampleStandardDeviation / Math.sqrt(sampleSize))
	const htypes = {0: '>', 1: '<', 2: '≠' }
	
	console.log(`
\n1. State assumptions: 
	* observations are independent
	* variable under study should be approximately normal
	* the variable under study should have no outliers
`)

	console.log('\n2. State hypotheses:')
	console.log(`H0: μ = ${hypothesisMean}`)
	console.log(`H1: μ ${htypes[hypothesisType]} ${hypothesisMean}`)

	console.log(`\n3. Significance value α = ` + significance)

	log('\n4. Compute test statistic t:')
	log('t = (x_ - u) / (s / sqrt(n))')
	console.log('t = ' + testStatistic)
	
	let p_value = 1 - TCDF(testStatistic, sampleSize - 1)


	log('\n5. Compute p-value for observing t:')
	log(`Use t-table to find P(t >= ${testStatistic}) = 1 - P(t <= ${testStatistic}):`)

	if (hypothesisType === 2) {
		p_value = p_value * 2
		log('Two tailed test so multiply p-value by two')
	}

	log(`p-value: ${p_value}`)

	const is_significant = p_value < significance

	if (is_significant) {
		console.log(`\n6. Evaluate: the result is significant as p = ${p_value} < α = ${significance}`)
	} else {
		console.log(`\n6. Evaluate: result is not significant`)
	}

	return {
		sampleSize,
		sampleMean,
		sampleStandardDeviation,
		hypothesisMean,
		significance,
		hypothesisType: htypes[hypothesisType],
		testStatistic,
		p_value,
		is_significant,
	}
}


function standardErrorMean() {
	return 'sqrt(variance / sampleSize)'
}

const sr = x => Math.sqrt(x)
const csv = x => window.x = CSV(x)[0]
const sd = v => Math.sqrt(x.svariance)

function sum(start, stop, f) {
	let s = 0

	for (let i=start; i <= stop; i++) {
		s += f(i)
	}

	return s
}

// logic section 

const logicNotes = {

	soundness: `
		A proof system is sound iff it cannot prove a contradiction
		|- p ==> |= p

		To prove soundness, use structural induction. Provde that |- p ==> |= p
		for basic formulas (using the chosen connectives), then deduce by induction
		on the length of a formula, that all provable formulas are valid.
		
	`,
	completeness: `
		A proof system is complete iff it can prove every true statement.
		|-p ==> |- p
	`,
	axiomaticProofSystems: `
		A proof system which holds one or more statements as basic. These 
		statements can be placed anywhere in a proof.
	`,
	firstOrderLanguage: `
		A first-order language is defined as L(C, F, P), where:

		* C is a set of constant symbols (nullary function symbols)
		* F is a set of function symbols each
		* P is a *non-empty* set of predicate symbols.
	`,
	firstOrderSyntax: `
		Let L(C, F, P) be a FOL. A well-formed formula in L is defined by the grammar
		below:

		
		Constant    ::= any member of C

		Function    ::= any member of F

		Predicate   ::= any member of P

		Variable    ::= any member of Var (a countable set of variables)

                Operator    ::= v
                              | ^
                              | ->
                              | <-->

		Quantifier  ::=

		Term        ::= Constant
			      | Variable
                              | Function ( Term, Term, ... )


		Atom        ::= Predicate ( Term, Term, ... ) 

		Unary       ::= ¬ Formula


         	Binary      ::= ( Formula Operator Formula )

                Quantified  ::= Quantifier Variable Formula

		Formula     ::= Atom
			      | Unary
             	    	      | Binary
			      | Quantified
	`,
	firstOrderSemantics: `
		For some first-order language L(C, F, P), an L-structure (or structure for L) is given by

		M = (D, I) where

		* D is a non-empty set refered to as the domain.
		* I is an interpretation function which gives meaning to symbols in L.

		  - The interpretation of a constant symbol is a member of D
		  - The interpretation of an n-ary function symbol is a function with signature D^n -> D
		  - The interpretation of an n-ary predicate symbol is a subset of D^n.

		A variable assignment (A), is a mapping between variable symbols and the domain.



		With these constructs we can now define how to evaluate WFFs. Let V be an evaluation 
		function over L. Let VMA[x] be the evaluation of x in L-structure M, under assignment A.

		Terms:

		(1) VMA[c] = I(c) where c is a Constant

		(2) VMA[v] = A(v) where v is a Variable

		(3) VMA[ f(t1, ..., tn) ] = I(f)(VMA[t1], ..., VMA[tn]) where f is a Function and ti are terms.


		Formulas:

		Let M|=A x denote "x is satisfiable in L-structure M under assignment A"

		(1) M|=A P(t1, ..., tn) <==> (VMA[t1], ..., VMA[tn]) << I(P)

		(2) M|=A ¬f <==> M|/=A f

		(3) M|=A (f1 ^ f2) <==> (M|=A f1) and (M|=A f2)

		(4) M|=A (f1 v f2) <==> (M|=A f1) or (M|=A f2)

		(5) M|=A (f1 -> f2) <==> (M|/=A f1) or (M|=A f2)

		(6) M|=A Ex f <==> M|=A[x -> d] f for some d in D. Here an x-variant of the assignment A is created
                    on the RHS so that f is evaluated with a *particular* member of the domain.


		Validity:

		* f is valid in structure M <==> for all assignments A, M|=A f      (written M|= f)
		* f is valid is general <==> for all structures M, f is valid in M  (written |= f)
	`,
	disjunctiveNormalForm: `
		A disjunction of conjunctive clauses. 

		Formally:

		DisjunctiveCaluse ::= ConjunctiveClause
				    | (ConjunctiveClause v DisjunctiveClause)

		ConjunctiveClause ::= Literal
				    | (Literal ^ ConjunctiveClause)

		Literal           ::= AtomicProposition
				    | ¬AtomProposition


		Tableaux convertion to DNF:

		* Construct a complete tableau for F
		* Take every open branch.

		* Let C(i) = "the conjunction of literals in open branch i"
		* Let D = "the disjunction of every C(i)"

		* Now D is the DNF equivalent of F. 
	`,
	herbrandStructure: `
		A first-order structure where:

		The domain consists only of closed/ground terms, constructed in the following way:

		D = {c1, ..., cn, f(c1), ..., f(cn), g(c1), ..., g(cn), fg(c1), ..., fg(cn), gf(c1), ..., gf(cn), ... etc}

		* The interpretation of a constant symbol is just that symbol. 
		* The interpretation of a function symbol is a function which takes n arguments and returns the closed term for that function.

		  For example, I(Add(1, 2)) = 'Add(1, 2)' where 'Add(1, 2)' is a member of the domain.

		In general:

		I(c) = c where c << C
	
		I(f) : (t1, ..., tn) -> 'f(t1, ..., tn)' where f << F

		I(p) can be chosen arbitrarily
	`,
	herbrandTheorem: `
		Let L be a first-order language with infinitely many constant symbols (and no equality).

		If p is satisfiable in some L-structure S, under assignment A, then p is also satisfiable
		n a Herbrand structure under assignment A.
	`,
	firstOrderEquality: `
		Usually equality = is treated as a primitive binary logical connective. The following
		inference rules hold:

		A(t), t = s |- A(s)	(substitution)

		A(t), s = t |- A(t)	(substitution)

		¬(t = t) |- #		(identity)
	`,
	proofNotation: `

		Definition of ! |- p:

		* Axiom systems: there is a finite sequence p0,p1,...,pk where each pi is either an axiom instance,
		  a member of !, or obtained from earlier formulas by rule of inference; and pk = p.

		* Tableaux: there is a finite closed tableau with ¬p at the root and constructed by alpha, beta, 
		  delta and gamma expansions, or by adding in nodes labelled by formulas in !.


		Definition of ! |= p:

		* For every L-structure S, if S is a model of ! (S |= !) then S is also a model of p (S |= p).


		Definition of inconsistancy:

		* If a set of L-formulas (Sigma) is inconsistent then Sigma |- #, i.e. the set leads to a
		  contradiction.

		
		Definition of soundness of |-:

		* |- is sound if ! |- p implies ! |= p. It follows that all inconsistent sets of sentences do not have
		  models.


		Definition of strongly complete :

		* Strongly complete means ! |= p implies ! |- p. It follows that all consistent sets of sentences
 		  have models. 
	`,
	compactnessTheorem: `

		* If for a theory (Sigma), Sigma |- p, then for at least one finite subset (Sigma0), Sigma0 |- p.

		* If Sigma is inconsistent, then by compactness, Sigma0 must be inconsistent. By strong completeness,
		  every consistent set has a model.

		  So if Sigma is inconsistent then there exists a Sigma0 without a model.
		  Conversely, if all subsets have a model, then Sigma is *consistent* and has a model as a whole.

		  In other words, it follows from compactness and completeness, that Sigma is consistent if and
		  only if every finite subset SigmaN is consistent.
	`,
	finiteness: `
		No first-order theory defines the class of all finite structures.
		
		Proof:

		(1) Suppose for contradiction that (D, I) |= F iff D is finite.
		(2) Let C be an infinite set of constants
		(3) Let Sigma = {¬(c = d) | c
	`,
	frameDefiningFormulas: `

		Property		Modal Formula		First-order Formula

		Reflexive		!p -> p			AwRww
		Transitive		!p -> !!p		AuAvAw((Ruv^Rvw)->Ruw)
		Symmetric		p -> !?p		AuAv(Ruv->Rvu)
		Dense			!!p->!p			AuAv(Ruv->Ew(Ruw^Rwv))
	`,
	labelledFrames: `
		A labelled Kripke frame is a Kripke frame with the addition of the labelling function
		(Lambda) which maps worlds to sets of formulas.
	`,
	modalTableau: `
		1. Make a queue of labelled frames.
		2. Start by enqueing a single frame with one world w, where w contains the root formula f.

		3. While the queue is not empty, dequeue a labelled frame ((W, R), L). W is the set of worlds,
		   E is the set of edges and L is the frame labelling function.

		3.1. If there is a world in W which contains a contradiction (i.e. {p, ¬p}), don't enqueue.

		3.2. Else if there are only boxed formulas, negated diamond formulas and literals in range(f), then
		     return model found.

		3.3. Else, pick a world v in W, and a formula p in the label of v. p must not be a literal, 
		     box formula or negated diamond.

		3.3.1. If p is an alpha-formula with expansions p1, p2, then in the label of v, replace p with
		       p1 and p2.

		3.3.2. Else if p is a beta-formula with expansions p1, p2, then create two new frames
		       ((W, R), L1) and ((W, R), L2). L1 and L2 are the same as L, except that the label of v
                       has p replaced with p1 and p2 respectively.

		3.3.3. Else if p is a diamond formula (<>q) or negated box formula (¬[]q), then add a new world u 
		       to W. 

		       > Create an edge from v to u. 
               > The label of u is the same as v, except:

               * u also contains q (if p is diamond)
		       * u also contains ¬q (if p is negated box)

		       * any box formulas []x in the label of v are replaced with x in u.
		       * any negated diamond formulas ¬<>x in the label of v are replaced with ¬x in u.

		4. Return no model found.



    Notes for reflexive frames:

            * Initially start with E = {(w, w)}.
            * And whenever a world v is added to W, also add (v, v) to E.
            * Whenever []A is added to the label of some world u, also add A
            * Whenever ¬<>A is added to the label of some world u, also add ¬A


	Notes for symmetric frames:

       * When handling diamond formulas, add an edge both ways (i.e. from u to v, but also from v to u).
	   * Any box or negated diamond formulas in in u should be back propagated to v. (I assume this means
         you strip the modal operator however).

	Notes for transitive frames:

	   * When handling a diamond formula in world v, after adding a new world u, and an edge (v, u), then
         also add an edge (x, u) whenever there is an existing edge (x, v). 

         This forms a transitive loop.

	   * Include A in the label of u whenever []A is in the label of x.

	   * Include ¬A in the label of u whenever ¬<>A is in the label of x. 
	`
}

// FOL stuff

class FirstOrderLanguage {

	constructor(constantSymbols, variableSymbols, functionSymbols, predicateSymbols, operatorSymbols='^v>') {

		this.constantSymbols = constantSymbols
		this.variableSymbols = variableSymbols
		this.functionSymbols = functionSymbols
		this.predicateSymbols = predicateSymbols
		this.operatorSymbols = operatorSymbols
	}
}


class FirstOrderNode {

	constructor(type, attributes) {
		Object.assign(this, { type }, attributes)
	}


	toString() {
		// TODO
	}
}


class FirstOrderParser {

	constructor(firstOrderLanguage) {

		this.firstOrderLanguage = firstOrderLanguage
		this.unconsumed = ''
	}


	use(string) {
		this.unconsumed = string
	}


	// Attempt to parse and return true if successful.

	is(parser, ...a) {

		const revertUnconsumed = this.unconsumed
		const yes = !!this.skip(parser, ...a)

		this.use(revertUnconsumed)
		return yes
	}


	// Attempt to parse and revert to previous unconsumed string on failure.

	skip(parser, ...a) {
		const revertUnconsumed = this.unconsumed

		try {
			return this[parser](...a)
		} catch {
			this.use(revertUnconsumed)
		}		
	}


	// Throw a syntax error due to a parsing failure.

	Failure(message) {
		throw new SyntaxError(message)
	}


	// Parse any of the specified characters.

	Match(characters) {
		const firstCharacter = this.unconsumed[0]

		if (this.unconsumed && characters.includes(firstCharacter)) {
			this.use(this.unconsumed.slice(1))

			return firstCharacter
		}

		return this.Failure(`Unexpected character ${firstCharacter}`)
	}


	// Parse terminals...

	LeftBrace() {
		return this.Match('(')
	}


	RightBrace() {
		return this.Match(')')
	}


	Comma() {
		return this.Match(',')
	}


	Quantifier() {
		return this.Match('AE')
	}


	UnaryOperator() {
		return this.Match('-')
	}


	Operator() {
		return this.Match(this.firstOrderLanguage.operatorSymbols)
	}


	ConstantSymbol() {
		return this.Match(this.firstOrderLanguage.constantSymbols)
	}


	VariableSymbol() {
		return this.Match(this.firstOrderLanguage.variableSymbols)
	}


	FunctionSymbol() {
		return this.Match(this.firstOrderLanguage.functionSymbols)
	}


	PredicateSymbol() {
		return this.Match(this.firstOrderLanguage.predicateSymbols)
	}


	// Parse literals...

	Constant() {
		return new FirstOrderNode('Constant', {
			constantSymbol: this.ConstantSymbol()
		})
	}


	Variable() {
		return new FirstOrderNode('Variable', {
			variableSymbol: this.VariableSymbol()
		})
	}


	Arguments() {
		const list = []

		this.LeftBrace()

		while (this.is('Literal')) {

			list.push(this.Literal())
			this.skip('Comma')
		}

		this.RightBrace()

		return new FirstOrderNode('Arguments', {
			arguments: list
		})

	}


	Function() {

		const functionSymbol = this.FunctionSymbol()
		const functionArguments = this.Arguments()		

		return new FirstOrderNode('Function', {
			functionSymbol,
			functionArguments
		})
	}


	Literal() {
		return this.skip('Constant')
			|| this.skip('Variable') 
			|| this.Function()
	}


	// Parse predicates.

	Predicate() {
		const predicateSymbol = this.PredicateSymbol()
		const predicateArguments = this.Arguments() 

		return new FirstOrderNode('Predicate', {
			predicateSymbol,
			predicateArguments
		})
	}


	// Parse primaries...

	Unary() {
		const operator = this.UnaryOperator()
		const formula = this.Formula()

		return new FirstOrderNode('Unary', {
			operator,
			formula
		})
	}


	Binary() {
		this.LeftBrace()

		const leftFormula = this.Formula()
		const operator = this.Operator()
		const rightFormula = this.Formula()

		this.RightBrace()

		return new FirstOrderNode('Binary', {
			operator,
			leftFormula,
			rightFormula
		})
	}


	Primary() {
		return this.skip('Unary') || this.Binary()
	}


	// Parse quantified formulas.

	Quantified() {
		const quantifier = this.Quantifier()
		const variable = this.Variable()
		const formula = this.Formula()

		return new FirstOrderNode('Quantified', {
			quantifier,
			variable,
			formula
		})		
	}


	// Parse formulas.

	Formula() {
		return this.skip('Predicate')
			|| this.skip('Primary')
			|| this.Quantified()
	}


	// Root parser.

	parse(string) {
		this.use(string)

		return this.Formula()
	}
}


class FirstOrderAssignment {

	constructor(variableMap={}) {
		this.variableMap = variableMap
	}

	set(variableSymbol, variableValue) {
		this.variableMap[variableSymbol] = variableValue	
	}

	get(variableSymbol) {
		return this.variableMap[variableSymbol]
	}
}


class FirstOrderStructure {

	constructor(firstOrderLanguage, domain, constantsMap, functionsMap, predicatesMap) {

		this.firstOrderLanguage = firstOrderLanguage
		this.domain = new Set(domain)
		this.interpretation = {...constantsMap, ...functionsMap, ...predicatesMap}
	}


	interpret(symbol) {
		return this.interpretation[symbol]
	}	
}


class FirstOrderEvaluator {

	constructor(firstOrderStructure, firstOrderAssignment) {

		this.firstOrderStructure = firstOrderStructure
		this.firstOrderAssignment = firstOrderAssignment ?? new FirstOrderAssignment()
	}


	evaluate(node) {

		if (node.type === 'Constant') return this.evaluateConstant(node)
		if (node.type === 'Variable') return this.evaluateVariable(node)
		if (node.type === 'Arguments') return this.evaluateArguments(node)
		if (node.type === 'Function') return this.evaluateFunction(node)
		if (node.type === 'Predicate') return this.evaluatePredicate(node)
		if (node.type === 'Unary') return this.evaluateUnary(node)
		if (node.type === 'Binary') return this.evaluateBinary(node)
		if (node.type === 'Quantified') return this.evaluateQuantified(node)
	}


	evaluateConstant(node) {
		return this.firstOrderStructure.interpret(node.constantSymbol)
	}


	evaluateVariable(node) {
		return this.firstOrderAssignment.get(node.variableSymbol)
	}


	evaluateArguments(node) {
		return node.arguments.map(this.evaluate.bind(this))
	}

	
	evaluateFunction(node) {

		const evaluatedFunction = this.firstOrderStructure.interpret(node.functionSymbol)
		const evaluatedArguments = this.evaluateArguments(node.functionArguments)

		return evaluatedFunction(...evaluatedArguments)
	}


	evaluatePredicate(node) {
		
		const evaluatedPredicate = this.firstOrderStructure.interpret(node.predicateSymbol)
		const evaluatedArguments = this.evaluateArguments(node.predicateArguments)

		return evaluatedPredicate(...evaluatedArguments)
	}


	evaluateUnary(node) {
		return !this.evaluate(node.formula)
	}

	
	evaluateBinary(node) {

		const { operator, leftFormula, rightFormula } = node
		const { operatorSymbols } = this.firstOrderStructure.firstOrderLanguage
		const [ conjunction, disjunction, implication ] = operatorSymbols

		if (operator === conjunction) return this.evaluate(leftFormula) && this.evaluate(rightFormula)
		if (operator === disjunction) return this.evaluate(leftFormula) || this.evaluate(rightFormula)
		if (operator === implication) return !this.evaluate(leftFormula) || this.evaluate(rightFormula)	
	}


	evaluateQuantified(node) {

		if (node.quantifier === 'E') return this.evaluateExistential(node)
		if (node.quantifier === 'A') return this.evaluateUniversal(node)
	}


	evaluateExistential(node) {

		for (let value of this.firstOrderStructure.domain) {
			
			this.firstOrderAssignment.set(node.variable.variableSymbol, value)
			
			if (this.evaluate(node.formula)) {
		
				this.existentialAssignment = {...this.firstOrderAssignment.variableMap}
				return true

			}
		}

		return false
	}


	evaluateUniversal(node) {

		for (let value of this.firstOrderStructure.domain) {

			this.firstOrderAssignment.set(node.variable.variableSymbol, value)

			if (! this.evaluate(node.formula)) {
		
				this.universalAssignment = {...this.firstOrderAssignment.variableMap}
				return false

			}
		}

		return true
	}
}


class FirstOrderLogic {

	constructor(domain, variables, constantsMap, functionsMap, predicatesMap) {

		const constants = Object.keys(constantsMap)
		const functions = Object.keys(functionsMap)
		const predicates = Object.keys(predicatesMap)
		const language = new FirstOrderLanguage(constants, variables, functions, predicates)
		const structure = new FirstOrderStructure(language, domain, constantsMap, functionsMap, predicatesMap)
		const parser = new FirstOrderParser(language)

		this.language = language
		this.structure = structure
		this.parser = parser
	}


	evaluate(string, assignment) {

		const evaluator = new FirstOrderEvaluator(this.structure, assignment)
		const node = this.parser.parse(string)

		return [evaluator.evaluate(node), evaluator]
	}
}


function mapDomain(domain) {
	return Object.fromEntries(domain.map(element => [element, element]))
}


folExample = new FirstOrderLogic(domain=[0,1,2,3,4], variableSymbols='xyz', constants=mapDomain([0,1,2,3,4]), functions={

    P(x, y) {
        return x + y
    },
    D(x, y) {
        return x - y
    },
    S(x) {
        return x + 1
    },
    N(x) {
        return x
    },
    I(x) {
        return -x
    }
},
predicates={
    L(x,y) {
        return x < y
    },
    G(x,y) {
        return x > y
    },
    M(x,y) {
        return x === y
    }
}
)


// graphs

class Graph {

	constructor() {
		this.nodes = {}
	}


	addNode(key, value=null) {

		this.nodes[key] = {
			value,
			edges: new Set()			
		}
	}


	removeNode(key) {
		delete this.nodes[key]
	}


	addEdge(leftKey, rightKey) {
		this.nodes[leftKey].edges.add(rightKey)
	}


	removeEdge(leftKey, rightKey) {
		this.nodes[leftKey].edges.delete(rightKey)
	}

	hasEdge(leftKey, rightKey) {
		return !!this.nodes[leftKey]?.edges.has(rightKey)
	}


	get keys() {
		return Object.keys(this.nodes)
	}
}


function createGraph(nodes, edges) {
	const graph = new Graph()

	nodes.forEach(key => graph.addNode(key))
	edges.forEach(keys => graph.addEdge(...keys))

	return graph
}


function createGraphFirstOrderModel(graph, predicatesMap={}, functionsMap={}, constantsMap={}, variables='abcdefgxyz') {

	// add the edge relation

	predicatesMap.C = function(x, y) {
		return graph.hasEdge(x, y)
	}

	// add the equality relation

	predicatesMap.M = function(x, y) {
		return x === y
	}


	// make the domain the set of graph nodes.

	const domain = graph.keys


	// create the model.

	const system = new FirstOrderLogic(domain, variables, constantsMap, functionsMap, predicatesMap)

	return system
}

Parsers.graphFromTuples = function(tuples) {

	const edges = tuples.split(/\)\s*,\s*/).map(pairString => pairString.replace('(', '').replace(')', ''))
		.map(pairString => pairString.split(/\s*,\s*/))
		.map(pair => [pair[0].replaceAll(/\W/g,''), pair[1].replaceAll(/\W/g, '')])

	const nodes = new Set(edges.join('').replaceAll(/\W/g, ''))

	return createGraph(nodes, edges)
}














// propositional logic

class PropositionalLanguage {

	constructor(variableSymbols='ABCDEFGHIJKLMNOPQRSTUVWXYZ', operatorSymbols='^v>') {

		this.variableSymbols = variableSymbols
		this.operatorSymbols = operatorSymbols
	}
}


class PropositionalNode {

	constructor(type, attributes) {
		Object.assign(this, { type }, attributes)
	}
}



class PropositionalParser {

	constructor(propositionalLanguage) {

		this.propositionalLanguage = propositionalLanguage
		this.unconsumed = ''
	}


	use(string) {
		this.unconsumed = string
	}


	// Attempt to parse and return true if successful.

	is(parser, ...a) {

		const revertUnconsumed = this.unconsumed
		const yes = !!this.skip(parser, ...a)

		this.use(revertUnconsumed)
		return yes
	}


	// Attempt to parse and revert to previous unconsumed string on failure.

	skip(parser, ...a) {
		const revertUnconsumed = this.unconsumed

		try {
			return this[parser](...a)
		} catch {
			this.use(revertUnconsumed)
		}		
	}


	// Throw a syntax error due to a parsing failure.

	Failure(message) {
		throw new SyntaxError(message)
	}


	// Parse any of the specified characters.

	Match(characters) {
		const firstCharacter = this.unconsumed[0]

		if (this.unconsumed && characters.includes(firstCharacter)) {
			this.use(this.unconsumed.slice(1))

			return firstCharacter
		}

		return this.Failure(`Unexpected character ${firstCharacter}`)
	}


	// Parse terminals...

	LeftBrace() {
		return this.Match('(')
	}


	RightBrace() {
		return this.Match(')')
	}


	UnaryOperator() {
		return this.Match('-')
	}


	Operator() {
		return this.Match(this.propositionalLanguage.operatorSymbols)
	}


	VariableSymbol() {
		return this.Match(this.propositionalLanguage.variableSymbols)
	}


	Proposition() {
		return new PropositionalNode('Proposition', {
			variableSymbol: this.VariableSymbol()
		})
	}


	Unary() {
		return new PropositionalNode('Unary', {
			operator: this.UnaryOperator(),
			formula: this.Formula()
		})
	}


	Binary() {
		this.LeftBrace()

		const leftFormula = this.Formula()
		const operator = this.Operator()
		const rightFormula = this.Formula()

		return new PropositionalNode('Binary', {
			operator,
			leftFormula,
			rightFormula
		})
	}


	Formula() {
		return this.skip('Proposition')
			|| this.skip('Unary')
			|| this.Binary()
	}


	parse(string) {
		this.use(string)

		return this.Formula()
	}
}


class PropositionalAssignment {

	constructor(variableMap={}) {
		this.variableMap = variableMap
	}

	set(variableSymbol, variableValue) {
		this.variableMap[variableSymbol] = variableValue	
	}

	get(variableSymbol) {
		return this.variableMap[variableSymbol]
	}
}



class PropositionalEvaluator {

	constructor(propositionalLanguage) {

		this.propositionalLanguage = propositionalLanguage
		this.propositionalAssignment = new PropositionalAssignment
	}


	assign(variableMap) {
		Object.assign(this.propositionalAssignment.variableMap, variableMap)
	}


	evaluate(node) {

		if (node.type === 'Proposition') return this.propositionalAssignment.get(node.variableSymbol)
		if (node.type === 'Unary') return !this.evaluate(node.formula)
		if (node.type === 'Binary') return this.evaluateBinary(node)
	}


	evaluateBinary({ operator, leftFormula, rightFormula }) {

		const [ conjunction, disjunction, implication ] = this.propositionalLanguage.operatorSymbols

		if (operator === conjunction) return this.evaluate(leftFormula) && this.evaluate(rightFormula)
		if (operator === disjunction) return this.evaluate(leftFormula) || this.evaluate(rightFormula)
		if (operator === implication) return !this.evaluate(leftFormula) || this.evaluate(rightFormula)
	}
}


class PropositionalLogic {

	constructor(variableSymbols) {

		const language = new PropositionalLanguage(variableSymbols)
		const parser = new PropositionalParser(language)
		const evaluator = new PropositionalEvaluator(language)

		this.langauge = language
		this.parser = parser
		this.evaluator = evaluator
	}


	evaluate(string, variableMap) {
		
		this.evaluator.assign(variableMap)

		const node = this.parser.parse(string)
		const value = this.evaluator.evaluate(node)

		return value
	}	
}








// modal logic

class ModalLanguage extends PropositionalLanguage {

	constructor(modalitySymbols='?!', variableSymbols, operatorSymbols) {

		super(variableSymbols, operatorSymbols)
		this.modalitySymbols = modalitySymbols
	}
}


class ModalParser extends PropositionalParser {

	ModalitySymbol() {
		return this.Match(this.propositionalLanguage.modalitySymbols)
	}


	Modality() {
		return new PropositionalNode('Modality', {

			modalitySymbol: this.ModalitySymbol(),
			formula: this.Formula()
		})
	}


	Formula() {
		return this.skip('Modality')
			|| this.skip('Proposition')
			|| this.skip('Unary')
			|| this.Binary()
	}
}


// Kripke semantics (yes this is just a copy and paste of Graph, since that's all a Kripke frame is).

class ModalFrame {

	constructor(keys, edges) {
		this.nodes = {}

		keys?.forEach(key => this.addNode(key))
		edges?.forEach(edge => this.addEdge(...edge))
	}


	addNode(key, value=null) {
		this.nodes[key] = new Set()
	}


	removeNode(key) {
		delete this.nodes[key]
	}


	addEdge(leftKey, rightKey) {
		this.nodes[leftKey].add(rightKey)
	}


	removeEdge(leftKey, rightKey) {
		this.nodes[leftKey].delete(rightKey)
	}

	hasEdge(leftKey, rightKey) {
		return !!this.nodes[leftKey]?.has(rightKey)
	}


	get keys() {
		return Object.keys(this.nodes)
	}
}


// Declares which variables are true in which worlds.
// Takes in a node key and returns a set of variable symbols.
// It differs from regular assignments only in regard to its domain and range.

class ModalAssignment extends PropositionalAssignment {

	get(perspectiveKey, variableSymbol) {
		return this.variableMap[variableSymbol]?.has(perspectiveKey)
	}
}


class ModalEvaluator {

	constructor(modalLanguage, modalFrame, modalAssignment) {

		this.modalLanguage = modalLanguage
		this.modalFrame = modalFrame
		this.modalAssignment = modalAssignment
	}


	evaluate(node, perspective) {

		if (node.type === 'Proposition') return this.modalAssignment.get(perspective, node.variableSymbol)
		if (node.type === 'Unary') return !this.evaluate(node.formula, perspective)
		if (node.type === 'Binary') return this.evaluateBinary(node, perspective)
		if (node.type === 'Modality') return this.evaluateModality(node, perspective)
	}


	evaluateBinary({ operator, leftFormula, rightFormula }, perspective) {

		const [ conjunction, disjunction, implication ] = this.modalLanguage.operatorSymbols

		if (operator === conjunction) return this.evaluate(leftFormula, perspective) && this.evaluate(rightFormula, perspective)
		if (operator === disjunction) return this.evaluate(leftFormula, perspective) || this.evaluate(rightFormula, perspective)
		if (operator === implication) return !this.evaluate(leftFormula, perspective) || this.evaluate(rightFormula, perspective)
	}


	evaluateModality({ modalitySymbol, formula }, perspective) {
		const [ possibility, neccessity ] = this.modalLanguage.modalitySymbols

		if (modalitySymbol === possibility) return this.evaluatePossibility(formula, perspective)
		if (modalitySymbol === neccessity) return this.evaluateNeccessity(formula, perspective)
	}


	evaluatePossibility(node, perspective) {

		for (let key of this.modalFrame.keys) {

			if (this.modalFrame.hasEdge(perspective, key)) {
				
				const value = this.evaluate(node, key)
				if (value) return true
			}
		}

		return false
	}


	evaluateNeccessity(node, perspective) {

		for (let key of this.modalFrame.keys) {

			if (this.modalFrame.hasEdge(perspective, key)) {

				const value = this.evaluate(node, key)
				if (!value) return false
			}
		}

		return true
	}
}


class ModalLogic { 

	constructor(nodes, edges, variableMap={}, variableSymbols, modalitySymbols) {

		this.language = new ModalLanguage(modalitySymbols, variableSymbols)
		this.parser = new ModalParser(this.language)
		this.frame = new ModalFrame(nodes, edges)
		this.assignment = new ModalAssignment(variableMap)
		this.evaluator = new ModalEvaluator(this.language, this.frame, this.assignment)
	}


	assign(variableMap) {
		Object.assign(this.assignment.variableMap, variableMap)
	}


	evaluate(string, perspectiveKey) {

		const node = this.parser.parse(string)
		const value = this.evaluator.evaluate(node, perspectiveKey)

		return value
	}
}


let modalExample = new ModalLogic(['X','Y','Z'], [['X','X'],['X', 'Y'],['X','Z'],['Z','Z']], {
    P: new Set(['X', 'Y']),
    Q: new Set(['X', 'Z']),
    R: new Set(['Z', 'Y']),
    S: new Set(['Z']),
    T: new Set(['Y'])
})


// herbrand structures

// the domain is the enumeration of all closed terms, i.e:
//
// {c1, ..., cn, f(c1), ..., f(cn), f1(c1), ...., f1(cn), ..., ff(c1), ..., ff(cn), ...}
// the 
/*

function *createHerbrandUniverse(constantSymbols, functionSymbols, functionArityMap) {

	const terms = [ ...constantSymbols ]

	for (let constantSymbol of constantSymbols) {
		yield constantSymbol
	}


	while (1) {

		for (let i=0; i < terms.length; i++) {

			
		}
	}	
}


function createHerbrandStructure(firstOrderLanguage, functionArityMap, predicatesMap) {

}*/


databaseNotes = {

	problemsWithFileSystemStorage: `
		- Isolation of data: each program maintains its own set of data and this data may not be
		  easy to share between programs.

		- Duplication of data: the same data may be held by different programs. This is a waste of
		  storage.

		- Data dependence: file structure is defined in a per-program manner.
	`,
	solutionsOfDatabases: `
		- Declarative: the way data is manipulated is not handled by programs. Programs only 
		  describe which changes they want.
	
		- data consistency
		- more information from the same amount of data.
		- improved data integrity
		- improved security 
	`,
	databaseDefinition: `
		A shared collection of logically related data desgned to meet the information needs of 
		an organisation.
	`,
	functionsOfADBMS: `
		* A data language
		* A security system
		* A data integrity system
		* A concurrency control system
		* A recovery control system
		* A user-accessible catalog
		
	`,
	normalisation: `
		* repeating group: an attribute which implicitly holds a nested table.

		* functional dependency: B is functionally dependent on A if knowing A allows us to uniquely determine B.

		* partial dependency: B is partially dependent on a *primary key* A, if B is functionally dependent on *part of* A.

		* transitive dependency: C is transitively dependent on A if B is functionally dependent on A and C is functionally dependent on B.

		* superkey - a set of attributes which together act as a primary key.

		* candidate key - a minimal superkey - i.e. removing one of the attributes would cause the 
                  combination to cease to act as a primary key. In other words it's a superkey that can't be made any
		  smaller.

		1NF:
			* no repeating groups.

		2NF:
			* must be in 1NF
			* no partial dependencies - every non-primary attribute is fully dependent on the primary key.

		3NF:
			* must be in 2NF
			* no non-primary attribute is transitively dependent on the primary key.

		BCNF:
			* must be in 3NF
			* for every dependency X --> Y, either Y is a subset of X or X is a superkey of the relation.

		4NF:

		5NF:	
	`	
	
}
