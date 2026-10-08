package test

import (
	"math"
	"testing"

	. "vectordb/pkg/vectordb"
)

func approxEqual(a, b, tol float32) bool {
	return math.Abs(float64(a-b)) <= float64(tol)
}

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name      string
		v1        []float32
		v2        []float32
		expected  float32
		expectErr bool
	}{
		{
			name:     "exact match",
			v1:       []float32{1.0, 2.0, 3.0},
			v2:       []float32{1.0, 2.0, 3.0},
			expected: 1.0,
		},
		{
			name:     "perpendicular",
			v1:       []float32{1.0, 0.0},
			v2:       []float32{0.0, 1.0},
			expected: 0.0,
		},
		{
			name:     "opposite directions",
			v1:       []float32{1.0, 0.0},
			v2:       []float32{-1.0, 0.0},
			expected: -1.0,
		},
		{
			name:      "dim mismatch",
			v1:        []float32{1.0, 2.0},
			v2:        []float32{1.0, 2.0, 3.0},
			expectErr: true,
		},
		{
			name:      "empty slices",
			v1:        []float32{},
			v2:        []float32{},
			expectErr: true,
		},
		{
			name:      "zero vector",
			v1:        []float32{0.0, 0.0},
			v2:        []float32{1.0, 2.0},
			expectErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := CosineSimilarity(tc.v1, tc.v2)
			if (err != nil) != tc.expectErr {
				t.Fatalf("got err=%v, wantErr=%v", err, tc.expectErr)
			}
			if !tc.expectErr && !approxEqual(got, tc.expected, 1e-5) {
				t.Errorf("got %v, want %v", got, tc.expected)
			}
		})
	}
}

func TestEuclideanDistance(t *testing.T) {
	tests := []struct {
		name      string
		v1        []float32
		v2        []float32
		expected  float32
		expectErr bool
	}{
		{
			name:     "identical points",
			v1:       []float32{1.0, 2.0, 3.0},
			v2:       []float32{1.0, 2.0, 3.0},
			expected: 0.0,
		},
		{
			name:     "3-4-5 right triangle",
			v1:       []float32{0.0, 0.0},
			v2:       []float32{3.0, 4.0},
			expected: 5.0,
		},
		{
			name:      "dimension mismatch",
			v1:        []float32{1.0},
			v2:        []float32{1.0, 2.0},
			expectErr: true,
		},
		{
			name:      "empty input",
			v1:        []float32{},
			v2:        []float32{},
			expectErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := EuclideanDistance(tc.v1, tc.v2)
			if (err != nil) != tc.expectErr {
				t.Fatalf("got err=%v, wantErr=%v", err, tc.expectErr)
			}
			if !tc.expectErr && !approxEqual(got, tc.expected, 1e-5) {
				t.Errorf("got %v, want %v", got, tc.expected)
			}
		})
	}
}
