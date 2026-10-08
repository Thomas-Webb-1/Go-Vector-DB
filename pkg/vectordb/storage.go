package vectordb

import (
	"errors"
	"sort"
	"sync"
)

var (
	ErrInvalidDimension = errors.New("vector dimension does not match database dimension")
	ErrHashNotFound     = errors.New("vector hash not found")
	ErrInvalidTopK       = errors.New("top k must be greater than zero")
	ErrUnsupportedMetric = errors.New("unsupported distance metric")
)

type VectorDatabase interface {
	Add(hash string, vector []float32) error
	Get(hash string) ([]float32, error)
	Remove(hash string) error
	Query(target []float32, metric MetricType, k int) ([]QueryResult, error)
	Count() int
	List() []VectorRecord
}

// Flat contiguous memory arena protected by RWMutex
// Avoids pointer chasing and heap fragmentation by packing all float32s into a single 1D buffer
type InMemoryVectorDB struct {
	mu        sync.RWMutex
	dim       int
	arena     []float32      // flat contiguous float buffer: [v0_0, v0_1, ..., v1_0, ...]
	index     map[string]int // hash -> slot index
	freeSlots []int          // recycled slot indices after deletion
}

func NewInMemoryVectorDB(dimension int) *InMemoryVectorDB {
	return &InMemoryVectorDB{
		dim:       dimension,
		arena:     make([]float32, 0, 1024*dimension),
		index:     make(map[string]int),
		freeSlots: make([]int, 0),
	}
}

func (db *InMemoryVectorDB) Add(hash string, vector []float32) error {
	if len(vector) != db.dim {
		return ErrInvalidDimension
	}

	db.mu.Lock()
	defer db.mu.Unlock()

	// check if key already exists, overwrite in place
	if slot, exists := db.index[hash]; exists {
		copy(db.arena[slot*db.dim:(slot+1)*db.dim], vector)
		return nil
	}

	// reuse freed slot to keep arena dense
	if len(db.freeSlots) > 0 {
		last := len(db.freeSlots) - 1
		slot := db.freeSlots[last]
		db.freeSlots = db.freeSlots[:last]

		db.index[hash] = slot
		copy(db.arena[slot*db.dim:(slot+1)*db.dim], vector)
		return nil
	}

	// append new vector to end of contiguous buffer
	slot := len(db.arena) / db.dim
	db.index[hash] = slot
	db.arena = append(db.arena, vector...)
	return nil
}

func (db *InMemoryVectorDB) Get(hash string) ([]float32, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	slot, ok := db.index[hash]
	if !ok {
		return nil, ErrHashNotFound
	}

	// copy out so caller cannot mutate our internal arena
	start := slot * db.dim
	cp := make([]float32, db.dim)
	copy(cp, db.arena[start:start+db.dim])
	return cp, nil
}

func (db *InMemoryVectorDB) Remove(hash string) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	slot, ok := db.index[hash]
	if !ok {
		return ErrHashNotFound
	}

	delete(db.index, hash)
	db.freeSlots = append(db.freeSlots, slot)
	return nil
}

func (db *InMemoryVectorDB) Count() int {
	db.mu.RLock()
	defer db.mu.RUnlock()
	return len(db.index)
}

func (db *InMemoryVectorDB) List() []VectorRecord {
	db.mu.RLock()
	defer db.mu.RUnlock()

	items := make([]VectorRecord, 0, len(db.index))
	for h, slot := range db.index {
		start := slot * db.dim
		cp := make([]float32, db.dim)
		copy(cp, db.arena[start:start+db.dim])
		items = append(items, VectorRecord{
			Hash:   h,
			Vector: cp,
		})
	}
	return items
}

func sortQueryResults(results []QueryResult, metric MetricType) {
	if metric == MetricCosineSimilarity {
		sort.Slice(results, func(i, j int) bool {
			return results[i].Score > results[j].Score
		})
	} else {
		sort.Slice(results, func(i, j int) bool {
			return results[i].Score < results[j].Score
		})
	}
}

type activeEntry struct {
	slot int
	hash string
}

func (db *InMemoryVectorDB) evalEntries(entries []activeEntry, target []float32, distFunc func([]float32, []float32) (float32, error)) ([]QueryResult, error) {
	res := make([]QueryResult, len(entries))
	for i, e := range entries {
		offset := e.slot * db.dim
		score, err := distFunc(target, db.arena[offset:offset+db.dim])
		if err != nil {
			return nil, err
		}
		res[i] = QueryResult{Hash: e.hash, Score: score}
	}
	return res, nil
}

func (db *InMemoryVectorDB) Query(target []float32, metric MetricType, k int) ([]QueryResult, error) {
	if len(target) != db.dim {
		return nil, ErrInvalidDimension
	}
	if k <= 0 {
		return nil, ErrInvalidTopK
	}

	var distFunc func([]float32, []float32) (float32, error)
	switch metric {
	case MetricCosineSimilarity:
		distFunc = CosineSimilarity
	case MetricEuclideanDistance:
		distFunc = EuclideanDistance
	default:
		return nil, ErrUnsupportedMetric
	}

	db.mu.RLock()
	defer db.mu.RUnlock()

	n := len(db.index)
	if n == 0 {
		return []QueryResult{}, nil
	}

	entries := make([]activeEntry, 0, n)
	for h, s := range db.index {
		entries = append(entries, activeEntry{slot: s, hash: h})
	}

	results, err := db.evalEntries(entries, target, distFunc)
	if err != nil {
		return nil, err
	}

	sortQueryResults(results, metric)

	if k > len(results) {
		k = len(results)
	}
	return results[:k], nil
}
