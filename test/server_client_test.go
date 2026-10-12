package test

import (
	"context"
	"testing"
	"time"

	. "vectordb/pkg/vectordb"
)

func TestServerClient_EndToEnd(t *testing.T) {
	db := NewInMemoryVectorDB(3)

	// let OS pick an open port
	srv := NewServer("127.0.0.1:0", db, nil)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go func() {
		_ = srv.Start(ctx)
	}()

	// brief pause so listener binds
	time.Sleep(30 * time.Millisecond)
	addr := srv.Addr().String()

	client := NewClient(addr, nil)
	if err := client.Connect(); err != nil {
		t.Fatalf("could not connect: %v", err)
	}
	defer client.Close()

	// populate with some 3d vectors
	if err := client.Add("vec_a", []float32{1.0, 0.0, 0.0}); err != nil {
		t.Fatalf("add a: %v", err)
	}
	if err := client.Add("vec_b", []float32{0.8, 0.2, 0.0}); err != nil {
		t.Fatalf("add b: %v", err)
	}
	if err := client.Add("vec_c", []float32{0.0, 1.0, 0.0}); err != nil {
		t.Fatalf("add c: %v", err)
	}

	// query for closest to [1, 0, 0]
	results, err := client.Query([]float32{1.0, 0.0, 0.0}, MetricCosineSimilarity, 2)
	if err != nil {
		t.Fatalf("query: %v", err)
	}

	if len(results) != 2 {
		t.Fatalf("expected 2 items, got %d", len(results))
	}
	if results[0].Hash != "vec_a" || results[1].Hash != "vec_b" {
		t.Errorf("unexpected top matches: %+v", results)
	}

	// delete top match and re-query
	if err := client.Remove("vec_a"); err != nil {
		t.Fatalf("remove: %v", err)
	}

	after, err := client.Query([]float32{1.0, 0.0, 0.0}, MetricCosineSimilarity, 1)
	if err != nil {
		t.Fatalf("query after del: %v", err)
	}
	if len(after) != 1 || after[0].Hash != "vec_b" {
		t.Errorf("expected vec_b after vec_a removal, got: %+v", after)
	}

	cancel()
	_ = srv.Stop()
}
