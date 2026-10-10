package test

import (
	"fmt"
	"math/rand"
	"testing"

	. "vectordb/pkg/vectordb"
)

func randVec(dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = rand.Float32()
	}
	return v
}

func BenchmarkDistanceMetrics(b *testing.B) {
	for _, dim := range []int{384, 768, 1536} {
		v1, v2 := randVec(dim), randVec(dim)

		b.Run(fmt.Sprintf("Cosine/dim=%d", dim), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				_, _ = CosineSimilarity(v1, v2)
			}
		})

		b.Run(fmt.Sprintf("Euclidean/dim=%d", dim), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				_, _ = EuclideanDistance(v1, v2)
			}
		})
	}
}

func BenchmarkDatabase_Add(b *testing.B) {
	dim := 1536
	vec := randVec(dim)
	db := NewInMemoryVectorDB(dim)

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = db.Add(fmt.Sprintf("%d", i), vec)
	}
}

func BenchmarkDatabase_Query(b *testing.B) {
	for _, dim := range []int{384, 768, 1536} {
		db := NewInMemoryVectorDB(dim)
		for i := 0; i < 1000; i++ {
			_ = db.Add(fmt.Sprintf("%d", i), randVec(dim))
		}
		target := randVec(dim)

		b.Run(fmt.Sprintf("Cosine/dim=%d/N=1000", dim), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				_, _ = db.Query(target, MetricCosineSimilarity, 10)
			}
		})

		b.Run(fmt.Sprintf("Euclidean/dim=%d/N=1000", dim), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				_, _ = db.Query(target, MetricEuclideanDistance, 10)
			}
		})
	}
}
