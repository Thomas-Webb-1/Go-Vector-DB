package test

import (
	"context"
	"fmt"
	"math/rand"
	"testing"
	"time"

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

func BenchmarkBinaryCodec(b *testing.B) {
	dim := 1536
	req := &Request{
		Type: MsgQuery,
		Query: &QueryPayload{
			Vector: randVec(dim),
			Metric: MetricCosineSimilarity,
			TopK:   10,
		},
	}

	codec := NewBinaryCodec()
	b.Run("Encode", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_, _ = codec.EncodeRequest(req)
		}
	})

	binBytes, _ := codec.EncodeRequest(req)

	b.Run("Decode", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_, _ = codec.DecodeRequest(binBytes)
		}
	})
}

func BenchmarkParallelClient_Query(b *testing.B) {
	dim := 128
	db := NewInMemoryVectorDB(dim)
	for i := 0; i < 300; i++ {
		_ = db.Add(fmt.Sprintf("%d", i), randVec(dim))
	}
	srv := NewServer("127.0.0.1:0", db, nil)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() { _ = srv.Start(ctx) }()

	for srv.Addr() == nil {
		time.Sleep(5 * time.Millisecond)
	}

	client := NewParallelClient(srv.Addr().String(), nil)
	_ = client.Connect()
	defer client.Close()

	target := randVec(dim)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = client.Query(target, MetricCosineSimilarity, 5)
	}
}

func BenchmarkDistributedClient_Query(b *testing.B) {
	dim := 128
	numServers := 3
	servers, _, addrs, cancel := setupTestCluster(&testing.T{}, numServers, dim)
	defer func() {
		cancel()
		for _, s := range servers {
			_ = s.Stop()
		}
	}()

	distClient, _ := NewDistributedClient(addrs, nil)
	_ = distClient.Connect()
	defer distClient.Close()

	for i := 0; i < 300; i++ {
		_ = distClient.Add(fmt.Sprintf("%d", i), randVec(dim))
	}

	target := randVec(dim)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = distClient.Query(target, MetricCosineSimilarity, 5)
	}
}
