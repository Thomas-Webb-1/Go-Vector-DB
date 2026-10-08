package vectordb

type MetricType string

const (
	MetricCosineSimilarity  MetricType = "COSINE"
	MetricEuclideanDistance MetricType = "EUCLIDEAN"
)

type VectorRecord struct {
	Hash   string
	Vector []float32
}

type QueryResult struct {
	Hash  string
	Score float32
}
