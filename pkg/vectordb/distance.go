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

	var dot, magA, magB float32
	for i := 0; i < n; i++ {
		vA, vB := a[i], b[i]
		dot += vA * vB
		magA += vA * vA
		magB += vB * vB
	}

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

	var sum float32
	for i := 0; i < n; i++ {
		diff := a[i] - b[i]
		sum += diff * diff
	}

	return float32(math.Sqrt(float64(sum))), nil
}
