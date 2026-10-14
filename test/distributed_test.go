package test

import (
	"context"
	"math"
	"testing"
	"time"

	. "vectordb/pkg/vectordb"
)

func setupTestCluster(t *testing.T, numServers int, dim int) ([]*Server, []*InMemoryVectorDB, []string, context.CancelFunc) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())

	servers := make([]*Server, numServers)
	dbs := make([]*InMemoryVectorDB, numServers)
	addrs := make([]string, numServers)

	for i := 0; i < numServers; i++ {
		db := NewInMemoryVectorDB(dim)
		srv := NewServer("127.0.0.1:0", db, nil)

		go func(s *Server) {
			_ = s.Start(ctx)
		}(srv)

		// Brief sleep to ensure listener is bound
		for j := 0; j < 50; j++ {
			if srv.Addr() != nil {
				break
			}
			time.Sleep(10 * time.Millisecond)
		}
		if srv.Addr() == nil {
			cancel()
			t.Fatalf("server %d failed to bind address", i)
		}

		servers[i] = srv
		dbs[i] = db
		addrs[i] = srv.Addr().String()
	}

	return servers, dbs, addrs, cancel
}

func TestAPIContract(t *testing.T) {
	// Verify at compile/runtime that both Parallel and Distributed clients implement the shared API contract
	var _ API = (*ParallelClient)(nil)
	var _ API = (*Client)(nil)
	var _ API = (*DistributedClient)(nil)
}

func TestDistributedClient_HashGetSetRemove(t *testing.T) {
	numServers := 3
	dim := 3
	servers, dbs, addrs, cancel := setupTestCluster(t, numServers, dim)
	defer func() {
		cancel()
		for _, s := range servers {
			_ = s.Stop()
		}
	}()

	distClient, err := NewDistributedClient(addrs, nil)
	if err != nil {
		t.Fatalf("failed to create distributed client: %v", err)
	}

	if err := distClient.Connect(); err != nil {
		t.Fatalf("failed to connect distributed client: %v", err)
	}
	defer distClient.Close()

	// Insert items
	keys := []string{"doc_apple", "doc_banana", "doc_cherry", "doc_date", "doc_elderberry", "doc_fig"}
	vectors := map[string][]float32{
		"doc_apple":      {1.0, 0.0, 0.0},
		"doc_banana":     {0.0, 1.0, 0.0},
		"doc_cherry":     {0.0, 0.0, 1.0},
		"doc_date":       {0.5, 0.5, 0.0},
		"doc_elderberry": {0.0, 0.5, 0.5},
		"doc_fig":        {0.3, 0.3, 0.3},
	}

	for _, k := range keys {
		if err := distClient.Add(k, vectors[k]); err != nil {
			t.Fatalf("failed to add %s: %v", k, err)
		}
	}

	// Verify each key was placed in its hashed shard's database
	for _, k := range keys {
		expectedShard := distClient.ShardIndex(k)
		shardDB := dbs[expectedShard]

		vec, err := shardDB.Get(k)
		if err != nil {
			t.Errorf("expected key %s in shard %d DB, but got error: %v", k, expectedShard, err)
		}
		if len(vec) != dim {
			t.Errorf("key %s vector dimension mismatch: got %d, want %d", k, len(vec), dim)
		}

		// Verify none of the other shard DBs contain this key
		for otherShard, otherDB := range dbs {
			if otherShard == expectedShard {
				continue
			}
			if _, err := otherDB.Get(k); err != ErrHashNotFound {
				t.Errorf("key %s should NOT exist in shard %d, but found or got err: %v", k, otherShard, err)
			}
		}
	}

	// Test Get via distributed client
	for _, k := range keys {
		vec, err := distClient.Get(k)
		if err != nil {
			t.Fatalf("distClient.Get(%s) failed: %v", k, err)
		}
		for d := 0; d < dim; d++ {
			if math.Abs(float64(vec[d]-vectors[k][d])) > 1e-6 {
				t.Errorf("distClient.Get(%s) mismatch at idx %d: got %f, want %f", k, d, vec[d], vectors[k][d])
			}
		}
	}

	// Test Count across cluster
	count, err := distClient.Count()
	if err != nil {
		t.Fatalf("distClient.Count() failed: %v", err)
	}
	if count != len(keys) {
		t.Errorf("distClient.Count() = %d, want %d", count, len(keys))
	}

	// Test List across cluster
	records, err := distClient.List()
	if err != nil {
		t.Fatalf("distClient.List() failed: %v", err)
	}
	if len(records) != len(keys) {
		t.Errorf("distClient.List() count = %d, want %d", len(records), len(keys))
	}

	// Test Remove
	toRemove := "doc_apple"
	expectedShard := distClient.ShardIndex(toRemove)
	if err := distClient.Remove(toRemove); err != nil {
		t.Fatalf("distClient.Remove(%s) failed: %v", toRemove, err)
	}

	// Verify key is gone from shard
	if _, err := dbs[expectedShard].Get(toRemove); err != ErrHashNotFound {
		t.Errorf("expected doc_apple to be removed from shard %d, got err: %v", expectedShard, err)
	}

	// Verify count decreased
	newCount, err := distClient.Count()
	if err != nil {
		t.Fatalf("distClient.Count() after remove failed: %v", err)
	}
	if newCount != len(keys)-1 {
		t.Errorf("newCount = %d, want %d", newCount, len(keys)-1)
	}
}

func TestDistributedClient_QueryScatterGather(t *testing.T) {
	numServers := 3
	dim := 2
	servers, _, addrs, cancel := setupTestCluster(t, numServers, dim)
	defer func() {
		cancel()
		for _, s := range servers {
			_ = s.Stop()
		}
	}()

	distClient, err := NewDistributedClient(addrs, nil)
	if err != nil {
		t.Fatalf("create dist client: %v", err)
	}
	if err := distClient.Connect(); err != nil {
		t.Fatalf("connect dist client: %v", err)
	}
	defer distClient.Close()

	// Seed vectors with known distances to query target [1.0, 0.0]
	testData := map[string][]float32{
		"v_best":  {1.0, 0.0},  // cosine = 1.0
		"v_good":  {0.9, 0.1},  // cosine ~ 0.99
		"v_mid":   {0.7, 0.7},  // cosine ~ 0.707
		"v_poor":  {0.1, 0.9},  // cosine ~ 0.11
		"v_worst": {-1.0, 0.0}, // cosine = -1.0
	}

	for h, vec := range testData {
		if err := distClient.Add(h, vec); err != nil {
			t.Fatalf("add %s: %v", h, err)
		}
	}

	// Query top 3 nearest by Cosine Similarity
	results, err := distClient.Query([]float32{1.0, 0.0}, MetricCosineSimilarity, 3)
	if err != nil {
		t.Fatalf("query cosine: %v", err)
	}

	if len(results) != 3 {
		t.Fatalf("expected 3 results, got %d", len(results))
	}

	if results[0].Hash != "v_best" {
		t.Errorf("expected top result 'v_best', got %s (score %f)", results[0].Hash, results[0].Score)
	}
	if results[1].Hash != "v_good" {
		t.Errorf("expected 2nd result 'v_good', got %s (score %f)", results[1].Hash, results[1].Score)
	}
	if results[2].Hash != "v_mid" {
		t.Errorf("expected 3rd result 'v_mid', got %s (score %f)", results[2].Hash, results[2].Score)
	}

	// Query top 2 nearest by Euclidean Distance
	euclidResults, err := distClient.Query([]float32{1.0, 0.0}, MetricEuclideanDistance, 2)
	if err != nil {
		t.Fatalf("query euclidean: %v", err)
	}

	if len(euclidResults) != 2 {
		t.Fatalf("expected 2 results, got %d", len(euclidResults))
	}
	if euclidResults[0].Hash != "v_best" {
		t.Errorf("expected top euclidean 'v_best', got %s", euclidResults[0].Hash)
	}
	if euclidResults[1].Hash != "v_good" {
		t.Errorf("expected 2nd euclidean 'v_good', got %s", euclidResults[1].Hash)
	}
}

func TestDistributedClient_Validation(t *testing.T) {
	// Empty address list error
	_, err := NewDistributedClient([]string{}, nil)
	if err != ErrNoServersConfigured {
		t.Errorf("expected ErrNoServersConfigured, got %v", err)
	}

	// Invalid TopK
	distClient, err := NewDistributedClient([]string{"127.0.0.1:9999"}, nil)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	_, err = distClient.Query([]float32{1.0, 2.0}, MetricCosineSimilarity, 0)
	if err != ErrInvalidTopK {
		t.Errorf("expected ErrInvalidTopK, got %v", err)
	}
}
