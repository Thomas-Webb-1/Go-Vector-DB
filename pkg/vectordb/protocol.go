package vectordb

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
)

type MessageType string

const (
	MsgAdd    MessageType = "ADD"
	MsgGet    MessageType = "GET"
	MsgRemove MessageType = "REMOVE"
	MsgQuery  MessageType = "QUERY"
	MsgCount  MessageType = "COUNT"
	MsgList   MessageType = "LIST"
)

type Request struct {
	Type   MessageType
	Add    *AddPayload
	Get    *GetPayload
	Remove *RemovePayload
	Query  *QueryPayload
}

type AddPayload struct {
	Hash   string
	Vector []float32
}

type GetPayload struct {
	Hash string
}

type RemovePayload struct {
	Hash string
}

type QueryPayload struct {
	Vector []float32
	Metric MetricType
	TopK   int
}

type Response struct {
	Success bool
	Error   string
	Vector  []float32
	Results []QueryResult
	Count   int
	Vectors []VectorRecord
}

type Codec interface {
	EncodeRequest(req *Request) ([]byte, error)
	DecodeRequest(data []byte) (*Request, error)
	EncodeResponse(res *Response) ([]byte, error)
	DecodeResponse(data []byte) (*Response, error)
}

func writeFrame(w io.Writer, payload []byte) error {
	buf := make([]byte, 4+len(payload))
	binary.BigEndian.PutUint32(buf[:4], uint32(len(payload)))
	copy(buf[4:], payload)
	_, err := w.Write(buf)
	return err
}

func readFrame(r io.Reader) ([]byte, error) {
	var hdr [4]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return nil, err
	}
	length := binary.BigEndian.Uint32(hdr[:])
	payload := make([]byte, length)
	if _, err := io.ReadFull(r, payload); err != nil {
		return nil, err
	}
	return payload, nil
}

// Binary Codec: zero-copy float packing without text/json overhead
const (
	opAdd    byte = 1
	opGet    byte = 2
	opRemove byte = 3
	opQuery  byte = 4
	opCount  byte = 5
	opList   byte = 6

	respFlagResults byte = 1 << 0
	respFlagVector  byte = 1 << 1
	respFlagCount   byte = 1 << 2
	respFlagVectors byte = 1 << 3
)

func appendStr(buf []byte, s string) []byte {
	buf = binary.LittleEndian.AppendUint16(buf, uint16(len(s)))
	return append(buf, s...)
}

func readStr(b []byte) (string, []byte, error) {
	if len(b) < 2 {
		return "", nil, errors.New("truncated string length")
	}
	l := int(binary.LittleEndian.Uint16(b[:2]))
	b = b[2:]
	if len(b) < l {
		return "", nil, errors.New("truncated string body")
	}
	return string(b[:l]), b[l:], nil
}

func appendFloats(buf []byte, v []float32) []byte {
	buf = binary.LittleEndian.AppendUint32(buf, uint32(len(v)))
	for _, f := range v {
		buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(f))
	}
	return buf
}

func readFloats(b []byte) ([]float32, []byte, error) {
	if len(b) < 4 {
		return nil, nil, errors.New("truncated float slice length")
	}
	n := int(binary.LittleEndian.Uint32(b[:4]))
	b = b[4:]
	if len(b) < n*4 {
		return nil, nil, errors.New("truncated float slice body")
	}
	v := make([]float32, n)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4 : (i+1)*4]))
	}
	return v, b[n*4:], nil
}

type BinaryCodec struct{}

func NewBinaryCodec() *BinaryCodec {
	return &BinaryCodec{}
}

func (b *BinaryCodec) EncodeRequest(req *Request) ([]byte, error) {
	buf := make([]byte, 0, 128)
	switch req.Type {
	case MsgAdd:
		if req.Add == nil {
			return nil, errors.New("missing add payload")
		}
		buf = append(buf, opAdd)
		buf = appendStr(buf, req.Add.Hash)
		buf = appendFloats(buf, req.Add.Vector)
		return buf, nil

	case MsgGet:
		if req.Get == nil {
			return nil, errors.New("missing get payload")
		}
		buf = append(buf, opGet)
		return appendStr(buf, req.Get.Hash), nil

	case MsgRemove:
		if req.Remove == nil {
			return nil, errors.New("missing remove payload")
		}
		buf = append(buf, opRemove)
		return appendStr(buf, req.Remove.Hash), nil

	case MsgQuery:
		if req.Query == nil {
			return nil, errors.New("missing query payload")
		}
		buf = append(buf, opQuery)
		metricByte := byte(1)
		if req.Query.Metric == MetricEuclideanDistance {
			metricByte = 2
		}
		buf = append(buf, metricByte)
		buf = binary.LittleEndian.AppendUint32(buf, uint32(req.Query.TopK))
		buf = appendFloats(buf, req.Query.Vector)
		return buf, nil

	case MsgCount:
		return append(buf, opCount), nil

	case MsgList:
		return append(buf, opList), nil

	default:
		return nil, errors.New("unknown request type")
	}
}

func (b *BinaryCodec) DecodeRequest(data []byte) (*Request, error) {
	if len(data) == 0 {
		return nil, errors.New("empty binary request")
	}
	op, r := data[0], data[1:]

	switch op {
	case opAdd:
		hash, r, err := readStr(r)
		if err != nil {
			return nil, fmt.Errorf("malformed add req key: %w", err)
		}
		vec, _, err := readFloats(r)
		if err != nil {
			return nil, fmt.Errorf("malformed add req vector: %w", err)
		}
		return &Request{Type: MsgAdd, Add: &AddPayload{Hash: hash, Vector: vec}}, nil

	case opGet:
		hash, _, err := readStr(r)
		if err != nil {
			return nil, fmt.Errorf("malformed get req key: %w", err)
		}
		return &Request{Type: MsgGet, Get: &GetPayload{Hash: hash}}, nil

	case opRemove:
		hash, _, err := readStr(r)
		if err != nil {
			return nil, fmt.Errorf("malformed remove req key: %w", err)
		}
		return &Request{Type: MsgRemove, Remove: &RemovePayload{Hash: hash}}, nil

	case opQuery:
		if len(r) < 5 {
			return nil, errors.New("malformed query req")
		}
		metric := MetricCosineSimilarity
		if r[0] == 2 {
			metric = MetricEuclideanDistance
		}
		topK := int(binary.LittleEndian.Uint32(r[1:5]))
		vec, _, err := readFloats(r[5:])
		if err != nil {
			return nil, fmt.Errorf("malformed query vector: %w", err)
		}
		return &Request{Type:  MsgQuery, Query: &QueryPayload{Vector: vec, Metric: metric, TopK: topK}}, nil

	case opCount:
		return &Request{Type: MsgCount}, nil

	case opList:
		return &Request{Type: MsgList}, nil

	default:
		return nil, errors.New("unknown opcode")
	}
}

func (b *BinaryCodec) EncodeResponse(res *Response) ([]byte, error) {
	buf := make([]byte, 0, 64)
	if !res.Success {
		buf = append(buf, 0)
		return appendStr(buf, res.Error), nil
	}

	buf = append(buf, 1) // success flag

	var flags byte
	if len(res.Results) > 0 {
		flags |= respFlagResults
	}
	if len(res.Vector) > 0 {
		flags |= respFlagVector
	}
	if res.Count > 0 {
		flags |= respFlagCount
	}
	if len(res.Vectors) > 0 {
		flags |= respFlagVectors
	}
	buf = append(buf, flags)

	if flags&respFlagResults != 0 {
		buf = binary.LittleEndian.AppendUint32(buf, uint32(len(res.Results)))
		for _, qr := range res.Results {
			buf = appendStr(buf, qr.Hash)
			buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(qr.Score))
		}
	}
	if flags&respFlagVector != 0 {
		buf = appendFloats(buf, res.Vector)
	}
	if flags&respFlagCount != 0 {
		buf = binary.LittleEndian.AppendUint32(buf, uint32(res.Count))
	}
	if flags&respFlagVectors != 0 {
		buf = binary.LittleEndian.AppendUint32(buf, uint32(len(res.Vectors)))
		for _, vr := range res.Vectors {
			buf = appendStr(buf, vr.Hash)
			buf = appendFloats(buf, vr.Vector)
		}
	}

	return buf, nil
}

func (b *BinaryCodec) DecodeResponse(data []byte) (*Response, error) {
	if len(data) == 0 {
		return nil, errors.New("empty response")
	}
	if data[0] == 0 { // failure
		errStr, _, _ := readStr(data[1:])
		if errStr == "" && len(data) <= 2 {
			errStr = "unknown error"
		}
		return &Response{Success: false, Error: errStr}, nil
	}

	res := &Response{Success: true}
	if len(data) < 2 {
		return res, nil
	}

	flags, r := data[1], data[2:]

	if flags&respFlagResults != 0 && len(r) >= 4 {
		num := int(binary.LittleEndian.Uint32(r[:4]))
		r = r[4:]
		results := make([]QueryResult, 0, num)
		for i := 0; i < num; i++ {
			h, next, err := readStr(r)
			if err != nil || len(next) < 4 {
				break
			}
			score := math.Float32frombits(binary.LittleEndian.Uint32(next[:4]))
			r = next[4:]
			results = append(results, QueryResult{Hash: h, Score: score})
		}
		res.Results = results
	}

	if flags&respFlagVector != 0 {
		if vec, next, err := readFloats(r); err == nil {
			res.Vector = vec
			r = next
		}
	}

	if flags&respFlagCount != 0 && len(r) >= 4 {
		res.Count = int(binary.LittleEndian.Uint32(r[:4]))
		r = r[4:]
	}

	if flags&respFlagVectors != 0 && len(r) >= 4 {
		num := int(binary.LittleEndian.Uint32(r[:4]))
		r = r[4:]
		records := make([]VectorRecord, 0, num)
		for i := 0; i < num; i++ {
			h, next, err := readStr(r)
			if err != nil {
				break
			}
			vec, next, err := readFloats(next)
			if err != nil {
				break
			}
			r = next
			records = append(records, VectorRecord{Hash: h, Vector: vec})
		}
		res.Vectors = records
	}

	return res, nil
}
