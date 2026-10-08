package test

import (
	"testing"

	. "vectordb/pkg/vectordb"
)

func TestInMemoryVectorDB_AddAndRemove(t *testing.T) {
	db := NewInMemoryVectorDB(3)

	// should accept valid 3D vector
	if err := db.Add("doc1", []float32{1.0, 0.0, 0.0}); err != nil {
		t.Fatalf("failed to insert doc1: %v", err)
	}
	if n := db.Count(); n != 1 {
		t.Errorf("wanted count 1, got %d", n)
	}

	// reject wrong dimensions
	if err := db.Add("doc2", []float32{1.0, 0.0}); err != ErrInvalidDimension {
		t.Errorf("expected ErrInvalidDimension, got %v", err)
	}

	// delete existing
	if err := db.Remove("doc1"); err != nil {
		t.Fatalf("remove failed: %v", err)
	}
	if n := db.Count(); n != 0 {
		t.Errorf("wanted count 0 after deletion, got %d", n)
	}

	// deleting again should error
	if err := db.Remove("doc1"); err != ErrHashNotFound {
		t.Errorf("expected ErrHashNotFound, got %v", err)
	}
}

func TestInMemoryVectorDB_QueryCosine(t *testing.T) {
	db := NewInMemoryVectorDB(2)

	_ = db.Add("identical", []float32{1.0, 0.0})
	_ = db.Add("similar", []float32{0.9, 0.1})
	_ = db.Add("orthogonal", []float32{0.0, 1.0})

	results, err := db.Query([]float32{1.0, 0.0}, MetricCosineSimilarity, 2)
	if err != nil {
		t.Fatalf("query failed: %v", err)
	}

	if len(results) != 2 {
		t.Fatalf("expected 2 matches, got %d", len(results))
	}
	if results[0].Hash != "identical" {
		t.Errorf("top match should be 'identical', got %q", results[0].Hash)
	}
	if results[1].Hash != "similar" {
		t.Errorf("2nd match should be 'similar', got %q", results[1].Hash)
	}
}

func TestInMemoryVectorDB_QueryEuclidean(t *testing.T) {
	db := NewInMemoryVectorDB(2)

	_ = db.Add("close", []float32{1.0, 1.0})
	_ = db.Add("mid", []float32{5.0, 5.0})
	_ = db.Add("far", []float32{10.0, 10.0})

	results, err := db.Query([]float32{1.1, 1.1}, MetricEuclideanDistance, 3)
	if err != nil {
		t.Fatalf("query failed: %v", err)
	}

	if len(results) != 3 {
		t.Fatalf("expected 3 matches, got %d", len(results))
	}
	if results[0].Hash != "close" || results[1].Hash != "mid" || results[2].Hash != "far" {
		t.Errorf("ordering mismatch: %+v", results)
	}
}
