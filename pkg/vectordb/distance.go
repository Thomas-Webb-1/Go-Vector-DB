package vectordb

import (
	"errors"
	"math"
)

var (
	ErrDimensionMismatch = errors.New("vector dimensions do not match")
	ErrEmptyVector       = errors.New("vector cannot be empty")
	ErrZeroMagnitude     = errors.New("cannot compute cosine similarity with zero magnitude vector")
)

// cosine distance
func CosineSimilarity(a, b []float32) (float32, error) {
	n := len(a)
	if n == 0 || len(b) == 0 {
		return 0, ErrEmptyVector
	}
	if n != len(b) {
		return 0, ErrDimensionMismatch
	}

	// unroll loop to allow the compiler to use simd
	var d0, d1, d2, d3 float32
	var ma0, ma1, ma2, ma3 float32
	var mb0, mb1, mb2, mb3 float32

	i := 0
	limit := n - (n % 4)
	for ; i < limit; i += 4 {
		vA0, vB0 := a[i], b[i]
		vA1, vB1 := a[i+1], b[i+1]
		vA2, vB2 := a[i+2], b[i+2]
		vA3, vB3 := a[i+3], b[i+3]

		d0 += vA0 * vB0
		d1 += vA1 * vB1
		d2 += vA2 * vB2
		d3 += vA3 * vB3

		ma0 += vA0 * vA0
		ma1 += vA1 * vA1
		ma2 += vA2 * vA2
		ma3 += vA3 * vA3

		mb0 += vB0 * vB0
		mb1 += vB1 * vB1
		mb2 += vB2 * vB2
		mb3 += vB3 * vB3
	}

	// finish any remaning calculations
	for ; i < n; i++ {
		vA, vB := a[i], b[i]
		d0 += vA * vB
		ma0 += vA * vA
		mb0 += vB * vB
	}

	dot := d0 + d1 + d2 + d3
	magA := ma0 + ma1 + ma2 + ma3
	magB := mb0 + mb1 + mb2 + mb3

	// avoid divide by zero if one is an all-zero vector
	if magA == 0 || magB == 0 {
		return 0, ErrZeroMagnitude
	}

	return float32(float64(dot) / (math.Sqrt(float64(magA)) * math.Sqrt(float64(magB)))), nil
}

// standard L2 distance
func EuclideanDistance(a, b []float32) (float32, error) {
	n := len(a)
	if n == 0 || len(b) == 0 {
		return 0, ErrEmptyVector
	}
	if n != len(b) {
		return 0, ErrDimensionMismatch
	}

	// using the same unrolling trick as in cosine
	var s0, s1, s2, s3 float32
	i := 0
	limit := n - (n % 4)

	for ; i < limit; i += 4 {
		diff0 := a[i] - b[i]
		diff1 := a[i+1] - b[i+1]
		diff2 := a[i+2] - b[i+2]
		diff3 := a[i+3] - b[i+3]

		s0 += diff0 * diff0
		s1 += diff1 * diff1
		s2 += diff2 * diff2
		s3 += diff3 * diff3
	}

	// finish any remaning calculations
	for ; i < n; i++ {
		diff := a[i] - b[i]
		s0 += diff * diff
	}

	sum := s0 + s1 + s2 + s3
	return float32(math.Sqrt(float64(sum))), nil
}
