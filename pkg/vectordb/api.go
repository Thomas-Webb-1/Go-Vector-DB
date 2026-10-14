package vectordb

// API is the shared contract implemented by both Parallel and Distributed database clients.
type API interface {
	Connect() error
	Close() error
	Add(hash string, vector []float32) error
	Get(hash string) ([]float32, error)
	Remove(hash string) error
	Query(target []float32, metric MetricType, k int) ([]QueryResult, error)
	Count() (int, error)
	List() ([]VectorRecord, error)
}


