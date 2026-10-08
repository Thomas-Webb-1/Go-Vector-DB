package vectordb

import (
	"sync"
)

// NaiveVectorDB is an initial prototype using map[string][]float32
// Preserved during early design before migrating to zero-alloc flat contiguous arena.
type NaiveVectorDB struct {
	mu      sync.RWMutex
	dim     int
	vectors map[string][]float32
}

func NewNaiveVectorDB(dimension int) *NaiveVectorDB {
	return &NaiveVectorDB{
		dim:     dimension,
		vectors: make(map[string][]float32),
	}
}

func (db *NaiveVectorDB) Add(hash string, vector []float32) error {
	if len(vector) != db.dim {
		return ErrInvalidDimension
	}
	db.mu.Lock()
	defer db.mu.Unlock()

	cp := make([]float32, db.dim)
	copy(cp, vector)
	db.vectors[hash] = cp
	return nil
}

func (db *NaiveVectorDB) Get(hash string) ([]float32, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	vec, ok := db.vectors[hash]
	if !ok {
		return nil, ErrHashNotFound
	}
	cp := make([]float32, db.dim)
	copy(cp, vec)
	return cp, nil
}

func (db *NaiveVectorDB) Remove(hash string) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if _, ok := db.vectors[hash]; !ok {
		return ErrHashNotFound
	}
	delete(db.vectors, hash)
	return nil
}

func (db *NaiveVectorDB) Count() int {
	db.mu.RLock()
	defer db.mu.RUnlock()
	return len(db.vectors)
}
