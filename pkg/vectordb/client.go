package vectordb

import (
	"errors"
	"fmt"
	"net"
	"sync"
)

type Client struct {
	addr  string
	codec Codec

	mu   sync.Mutex
	conn net.Conn
}

func NewClient(addr string, codec Codec) *Client {
	if codec == nil {
		codec = NewBinaryCodec()
	}
	return &Client{
		addr:  addr,
		codec: codec,
	}
}

func (c *Client) Connect() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	conn, err := net.Dial("tcp", c.addr)
	if err != nil {
		return fmt.Errorf("dial %s: %w", c.addr, err)
	}

	c.conn = conn
	return nil
}

func (c *Client) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.conn != nil {
		return c.conn.Close()
	}
	return nil
}

// sends a request and waits for length-prefixed response
func (c *Client) send(req *Request) (*Response, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.conn == nil {
		return nil, errors.New("client not connected")
	}

	data, err := c.codec.EncodeRequest(req)
	if err != nil {
		return nil, fmt.Errorf("encode req: %w", err)
	}

	if err := writeFrame(c.conn, data); err != nil {
		return nil, fmt.Errorf("write msg: %w", err)
	}

	respBuf, err := readFrame(c.conn)
	if err != nil {
		return nil, fmt.Errorf("read resp: %w", err)
	}

	resp, err := c.codec.DecodeResponse(respBuf)
	if err != nil {
		return nil, fmt.Errorf("decode resp: %w", err)
	}

	if !resp.Success {
		return nil, errors.New(resp.Error)
	}

	return resp, nil
}

func (c *Client) Add(hash string, vector []float32) error {
	_, err := c.send(&Request{
		Type: MsgAdd,
		Add: &AddPayload{
			Hash:   hash,
			Vector: vector,
		},
	})
	return err
}

func (c *Client) Get(hash string) ([]float32, error) {
	resp, err := c.send(&Request{
		Type: MsgGet,
		Get: &GetPayload{
			Hash: hash,
		},
	})
	if err != nil {
		return nil, err
	}
	return resp.Vector, nil
}

func (c *Client) Remove(hash string) error {
	_, err := c.send(&Request{
		Type: MsgRemove,
		Remove: &RemovePayload{
			Hash: hash,
		},
	})
	return err
}

func (c *Client) Query(vector []float32, metric MetricType, k int) ([]QueryResult, error) {
	resp, err := c.send(&Request{
		Type: MsgQuery,
		Query: &QueryPayload{
			Vector: vector,
			Metric: metric,
			TopK:   k,
		},
	})
	if err != nil {
		return nil, err
	}
	return resp.Results, nil
}

func (c *Client) Count() (int, error) {
	resp, err := c.send(&Request{
		Type: MsgCount,
	})
	if err != nil {
		return 0, err
	}
	return resp.Count, nil
}

func (c *Client) List() ([]VectorRecord, error) {
	resp, err := c.send(&Request{
		Type: MsgList,
	})
	if err != nil {
		return nil, err
	}
	return resp.Vectors, nil
}

// ParallelClient is an alias for Client, representing the single-server parallel API.
type ParallelClient = Client

// NewParallelClient creates a client connecting to a parallel VectorDB server.
func NewParallelClient(addr string, codec Codec) *ParallelClient {
	return NewClient(addr, codec)
}
