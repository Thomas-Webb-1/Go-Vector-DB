package vectordb

import (
	"errors"
	"fmt"
	"hash/fnv"
	"sync"
)

var (
	ErrNoServersConfigured = errors.New("distributed client requires at least one server address")
)

// Hasher is a function type that maps a string key to a 32-bit unsigned integer.
type Hasher func(key string) uint32

// DefaultHasher computes a 32-bit FNV-1a hash.
func DefaultHasher(key string) uint32 {
	h := fnv.New32a()
	_, _ = h.Write([]byte(key))
	return h.Sum32()
}

// DistributedClient distributes vector storage and search across multiple
// parallel VectorDB servers using hash partitioning for get/set/remove and
// scatter-gather map-reduce for vector queries.
type DistributedClient struct {
	mu      sync.RWMutex
	addrs   []string
	clients []*Client
	codec   Codec
	hasher  Hasher
}

// Compile-time interface checks
var _ API = (*Client)(nil)
var _ API = (*DistributedClient)(nil)

// NewDistributedClient creates a DistributedClient connected to multiple parallel servers.
func NewDistributedClient(addrs []string, codec Codec) (*DistributedClient, error) {
	if len(addrs) == 0 {
		return nil, ErrNoServersConfigured
	}
	if codec == nil {
		codec = NewBinaryCodec()
	}

	addrCopy := make([]string, len(addrs))
	copy(addrCopy, addrs)

	clients := make([]*Client, len(addrCopy))
	for i, addr := range addrCopy {
		clients[i] = NewClient(addr, codec)
	}

	return &DistributedClient{
		addrs:   addrCopy,
		clients: clients,
		codec:   codec,
		hasher:  DefaultHasher,
	}, nil
}

// SetHasher configures a custom key hashing function.
func (d *DistributedClient) SetHasher(hasher Hasher) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if hasher != nil {
		d.hasher = hasher
	}
}

// Addrs returns a copy of the configured backend server addresses.
func (d *DistributedClient) Addrs() []string {
	d.mu.RLock()
	defer d.mu.RUnlock()
	out := make([]string, len(d.addrs))
	copy(out, d.addrs)
	return out
}

// ShardIndex returns the server index responsible for the given key/hash.
func (d *DistributedClient) ShardIndex(hash string) int {
	d.mu.RLock()
	defer d.mu.RUnlock()
	n := uint32(len(d.clients))
	return int(d.hasher(hash) % n)
}

// ShardAddr returns the server address responsible for the given key/hash.
func (d *DistributedClient) ShardAddr(hash string) string {
	idx := d.ShardIndex(hash)
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.addrs[idx]
}

// Connect establishes connections to all backend parallel servers concurrently.
func (d *DistributedClient) Connect() error {
	d.mu.Lock()
	defer d.mu.Unlock()

	type connResult struct {
		idx int
		err error
	}
	ch := make(chan connResult, len(d.clients))
	for i, c := range d.clients {
		go func(idx int, client *Client) {
			ch <- connResult{idx: idx, err: client.Connect()}
		}(i, c)
	}

	var firstErr error
	for i := 0; i < len(d.clients); i++ {
		res := <-ch
		if res.err != nil && firstErr == nil {
			firstErr = fmt.Errorf("failed to connect to shard %s (index %d): %w", d.addrs[res.idx], res.idx, res.err)
		}
	}

	if firstErr != nil {
		for _, c := range d.clients {
			_ = c.Close()
		}
		return firstErr
	}

	return nil
}

// Close terminates connections to all backend parallel servers.
func (d *DistributedClient) Close() error {
	d.mu.Lock()
	defer d.mu.Unlock()

	var firstErr error
	for _, c := range d.clients {
		if err := c.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

func (d *DistributedClient) shardClient(hash string) *Client {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.clients[int(d.hasher(hash)%uint32(len(d.clients)))]
}

func (d *DistributedClient) getClients() []*Client {
	d.mu.RLock()
	defer d.mu.RUnlock()
	clients := make([]*Client, len(d.clients))
	copy(clients, d.clients)
	return clients
}

// Add routes the vector to the designated shard based on the hash of its key.
func (d *DistributedClient) Add(hash string, vector []float32) error {
	return d.shardClient(hash).Add(hash, vector)
}

// Get retrieves the vector from the designated shard based on the hash of its key.
func (d *DistributedClient) Get(hash string) ([]float32, error) {
	return d.shardClient(hash).Get(hash)
}

// Remove deletes the vector from the designated shard based on the hash of its key.
func (d *DistributedClient) Remove(hash string) error {
	return d.shardClient(hash).Remove(hash)
}

// Query performs a distributed scatter-gather nearest neighbor search across all shards.
func (d *DistributedClient) Query(target []float32, metric MetricType, k int) ([]QueryResult, error) {
	if k <= 0 {
		return nil, ErrInvalidTopK
	}

	clients := d.getClients()
	type queryResult struct {
		results []QueryResult
		err     error
	}

	ch := make(chan queryResult, len(clients))
	for _, c := range clients {
		go func(client *Client) {
			res, err := client.Query(target, metric, k)
			ch <- queryResult{results: res, err: err}
		}(c)
	}

	var combined []QueryResult
	var firstErr error
	for i := 0; i < len(clients); i++ {
		qr := <-ch
		if qr.err != nil && firstErr == nil {
			firstErr = qr.err
		}
		if qr.err == nil {
			combined = append(combined, qr.results...)
		}
	}

	if firstErr != nil {
		return nil, firstErr
	}

	sortQueryResults(combined, metric)
	if k > len(combined) {
		k = len(combined)
	}

	return combined[:k], nil
}

// Count queries all shards concurrently and aggregates the total vector count.
func (d *DistributedClient) Count() (int, error) {
	clients := d.getClients()
	type countResult struct {
		count int
		err   error
	}

	ch := make(chan countResult, len(clients))
	for _, c := range clients {
		go func(client *Client) {
			cnt, err := client.Count()
			ch <- countResult{count: cnt, err: err}
		}(c)
	}

	total := 0
	var firstErr error
	for i := 0; i < len(clients); i++ {
		res := <-ch
		if res.err != nil && firstErr == nil {
			firstErr = res.err
		}
		total += res.count
	}

	if firstErr != nil {
		return 0, firstErr
	}

	return total, nil
}

// List queries all shards concurrently and returns the combined list of all vector records.
func (d *DistributedClient) List() ([]VectorRecord, error) {
	clients := d.getClients()
	type listResult struct {
		vectors []VectorRecord
		err     error
	}

	ch := make(chan listResult, len(clients))
	for _, c := range clients {
		go func(client *Client) {
			vecs, err := client.List()
			ch <- listResult{vectors: vecs, err: err}
		}(c)
	}

	var all []VectorRecord
	var firstErr error
	for i := 0; i < len(clients); i++ {
		res := <-ch
		if res.err != nil && firstErr == nil {
			firstErr = res.err
		}
		all = append(all, res.vectors...)
	}

	if firstErr != nil {
		return nil, firstErr
	}

	return all, nil
}

