package test

import (
	"reflect"
	"testing"

	. "vectordb/pkg/vectordb"
)


func TestBinaryCodec_RequestRoundTrip(t *testing.T) {
	codec := NewBinaryCodec()

	tests := []struct {
		name string
		req  *Request
	}{
		{
			name: "add command",
			req: &Request{
				Type: MsgAdd,
				Add: &AddPayload{
					Hash:   "doc_123",
					Vector: []float32{0.1, 0.2, 0.3},
				},
			},
		},
		{
			name: "get command",
			req: &Request{
				Type: MsgGet,
				Get:  &GetPayload{Hash: "doc_123"},
			},
		},
		{
			name: "count command",
			req: &Request{
				Type: MsgCount,
			},
		},
		{
			name: "list command",
			req: &Request{
				Type: MsgList,
			},
		},
		{
			name: "remove command",
			req: &Request{
				Type:   MsgRemove,
				Remove: &RemovePayload{Hash: "doc_123"},
			},
		},
		{
			name: "query command",
			req: &Request{
				Type: MsgQuery,
				Query: &QueryPayload{
					Vector: []float32{0.4, 0.5, 0.6},
					Metric: MetricCosineSimilarity,
					TopK:   5,
				},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			b, err := codec.EncodeRequest(tc.req)
			if err != nil {
				t.Fatalf("encode failed: %v", err)
			}

			out, err := codec.DecodeRequest(b)
			if err != nil {
				t.Fatalf("decode failed: %v", err)
			}

			if !reflect.DeepEqual(tc.req, out) {
				t.Errorf("mismatch: want %+v, got %+v", tc.req, out)
			}
		})
	}
}

func TestBinaryCodec_ResponseRoundTrip(t *testing.T) {
	codec := NewBinaryCodec()

	tests := []struct {
		name string
		res  *Response
	}{
		{
			name: "ok with matches",
			res: &Response{
				Success: true,
				Results: []QueryResult{
					{Hash: "doc_1", Score: 0.98},
					{Hash: "doc_2", Score: 0.85},
				},
			},
		},
		{
			name: "ok with get vector and list",
			res: &Response{
				Success: true,
				Vector:  []float32{1.0, 2.0, 3.0},
				Count:   2,
				Vectors: []VectorRecord{
					{Hash: "v1", Vector: []float32{1.0, 2.0, 3.0}},
					{Hash: "v2", Vector: []float32{4.0, 5.0, 6.0}},
				},
			},
		},
		{
			name: "error payload",
			res: &Response{
				Success: false,
				Error:   "key not found",
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			b, err := codec.EncodeResponse(tc.res)
			if err != nil {
				t.Fatalf("encode failed: %v", err)
			}

			out, err := codec.DecodeResponse(b)
			if err != nil {
				t.Fatalf("decode failed: %v", err)
			}

			if !reflect.DeepEqual(tc.res, out) {
				t.Errorf("mismatch: want %+v, got %+v", tc.res, out)
			}
		})
	}
}

