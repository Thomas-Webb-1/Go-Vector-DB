# In-Memory Distributed Vector Database

A high-performance, low-latency in-memory vector database implemented in Go. Built from scratch with flat memory arenas, instruction-level parallelism (ILP) loop unrolling, a custom binary wire protocol over TCP, and distributed scatter-gather sharding.

---

## Performance Optimisations

Nearest-neighbor search across millions of dimensions is memory- and throughput-bound. Key engineering optimizations include:

1. **Flat Contiguous Memory Arena (`storage.go`)**
   Standard slice-of-slices allocations (`[][]float32`) introduce pointer indirection and GC overhead. Packing embeddings into a contiguous 1D slice (`arena []float32`) allows the CPU to stream vectors through L1/L2 cache lines with near-zero cache misses and **0 B/op** allocations during queries.

2. **4-Way Loop-Unrolled Distance Kernels (`distance.go`)**
   Vector distance calculations (Cosine, Euclidean) are unrolled 4-way using independent accumulators (`s0, s1, s2, s3`). This breaks serial dependency chains, allowing out-of-order CPU pipelines to execute multiple floating-point multiply-adds in parallel via SIMD.

3. **Custom Binary Wire Framing (`protocol.go`)**
   Serializing high-dimensional float arrays with JSON creates severe CPU and garbage collection bottlenecks. The custom length-prefixed binary frame decodes in **~7.3 µs** with only 3 heap allocations (6.2 KB) for 1536-dimensional query vectors.

---

## Benchmark Highlights

Evaluated across standard embedding dimensions (`goos: linux`, `goarch: arm64`, Go 1.22):

- **Vector Distance Kernels (Zero-Allocation):**
  - 384-dim: Euclidean **211.4 ns/op** | Cosine **260.1 ns/op** (0 B/op, 0 allocs)
  - 768-dim: Euclidean **511.2 ns/op** | Cosine **492.4 ns/op** (0 B/op, 0 allocs)
  - 1536-dim: Euclidean **841.5 ns/op** | Cosine **973.8 ns/op** (0 B/op, 0 allocs)
- **Protocol Binary Codec (1536-dim frame):**
  - Decode: **~7.3 µs** (7,307 ns/op, 6,240 B/op, 3 allocs)
  - Encode: **~16.0 µs** (15,955 ns/op, 24,704 B/op, 10 allocs)
- **End-to-End Query Latency (over TCP, $k=5$):**
  - Single-node TCP (`ParallelClient`): **~441.5 µs**
  - 3-node distributed cluster scatter-gather (`DistributedClient`): **~438.5 µs**
- **In-Memory Linear Scan Search ($N=1000$ vectors, $k=10$):**
  - 384-dim: **~535.3 µs** (Cosine) / **~545.6 µs** (Euclidean)
  - 768-dim: **~974.8 µs** (Cosine) / **~902.6 µs** (Euclidean)
  - 1536-dim: **~843.3 µs** (Cosine) / **~1.27 ms** (Euclidean)

### Raw Benchmark Suite Output

```text
goos: linux
goarch: arm64
pkg: vectordb/test
BenchmarkDistanceMetrics/Cosine/dim=384-12               4195003               260.1 ns/op             0 B/op          0 allocs/op
BenchmarkDistanceMetrics/Euclidean/dim=384-12            6109929               211.4 ns/op             0 B/op          0 allocs/op
BenchmarkDistanceMetrics/Cosine/dim=768-12               2354877               492.4 ns/op             0 B/op          0 allocs/op
BenchmarkDistanceMetrics/Euclidean/dim=768-12            2815448               511.2 ns/op             0 B/op          0 allocs/op
BenchmarkDistanceMetrics/Cosine/dim=1536-12              1242248               973.8 ns/op             0 B/op          0 allocs/op
BenchmarkDistanceMetrics/Euclidean/dim=1536-12           1468552               841.5 ns/op             0 B/op          0 allocs/op
BenchmarkDatabase_Add-12                                   44355             25084 ns/op           31118 B/op          2 allocs/op
BenchmarkDatabase_Query/Cosine/dim=384/N=1000-12            2554            535291 ns/op           76168 B/op         47 allocs/op
BenchmarkDatabase_Query/Euclidean/dim=384/N=1000-12         2605            545588 ns/op           76119 B/op         47 allocs/op
BenchmarkDatabase_Query/Cosine/dim=768/N=1000-12            1044            974831 ns/op           76120 B/op         47 allocs/op
BenchmarkDatabase_Query/Euclidean/dim=768/N=1000-12         1138            902628 ns/op           76118 B/op         47 allocs/op
BenchmarkDatabase_Query/Cosine/dim=1536/N=1000-12           1218            843297 ns/op           76127 B/op         47 allocs/op
BenchmarkDatabase_Query/Euclidean/dim=1536/N=1000-12        1443           1273983 ns/op           76120 B/op         47 allocs/op
BenchmarkBinaryCodec/Encode-12                             68186             15955 ns/op           24704 B/op         10 allocs/op
BenchmarkBinaryCodec/Decode-12                            202819              7307 ns/op            6240 B/op          3 allocs/op
BenchmarkParallelClient_Query-12                            3784            441500 ns/op           30662 B/op         71 allocs/op
BenchmarkDistributedClient_Query-12                         2870            438535 ns/op           44016 B/op        229 allocs/op
PASS
ok      vectordb/test   31.585s
```

---

## Project Structure

```text
vectordb/
├── pkg/vectordb/
│   ├── api.go          # Common API interface for local, parallel, and distributed engines
│   ├── client.go       # TCP client with connection reuse and buffered streaming
│   ├── distance.go     # Unrolled distance kernels (Cosine, Euclidean, Dot Product)
│   ├── distributed.go  # FNV-1a sharding, coordinator, and scatter-gather query aggregation
│   ├── protocol.go     # Length-prefixed binary wire protocol and codecs
│   ├── server.go       # Non-blocking TCP server and request dispatcher
│   ├── storage.go      # Flat contiguous arena storage with free-list slot recycling
│   └── types.go        # Core data structures and metric definitions
└── test/               # Comprehensive unit, cluster, and performance benchmark suites
```

---

## Running & Testing

Run all unit and integration tests:

```bash
go test -v ./test/...
```

Run performance benchmarks with memory allocation metrics:

```bash
go test -run='^$' -bench='.' -benchmem vectordb/test
```
