package vectordb

import (
	"context"
	"fmt"
	"net"
	"sync"
)

type Server struct {
	addr     string
	listener net.Listener
	db       VectorDatabase
	codec    Codec

	mu     sync.Mutex
	closed bool
	wg     sync.WaitGroup
}

func NewServer(addr string, db VectorDatabase, codec Codec) *Server {
	if codec == nil {
		codec = NewBinaryCodec()
	}
	return &Server{
		addr:  addr,
		db:    db,
		codec: codec,
	}
}

// Addr returns the bound address (useful in tests when port is :0)
func (s *Server) Addr() net.Addr {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.listener != nil {
		return s.listener.Addr()
	}
	return nil
}

func (s *Server) Start(ctx context.Context) error {
	ln, err := net.Listen("tcp", s.addr)
	if err != nil {
		return fmt.Errorf("listen %s: %w", s.addr, err)
	}

	s.mu.Lock()
	s.listener = ln
	s.mu.Unlock()

	// stop accepting when context is cancelled
	go func() {
		<-ctx.Done()
		_ = s.Stop()
	}()

	for {
		conn, err := ln.Accept()
		if err != nil {
			s.mu.Lock()
			closed := s.closed
			s.mu.Unlock()
			if closed {
				return nil
			}
			return err
		}

		s.wg.Add(1)
		go func(c net.Conn) {
			defer s.wg.Done()
			s.handleConn(c)
		}(conn)
	}
}

func (s *Server) Stop() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	var err error
	if s.listener != nil {
		err = s.listener.Close()
	}
	s.mu.Unlock()

	return err
}

func (s *Server) handleConn(conn net.Conn) {
	defer conn.Close()

	for {
		payload, err := readFrame(conn)
		if err != nil {
			return
		}
		if len(payload) == 0 {
			continue
		}

		req, err := s.codec.DecodeRequest(payload)
		var resp *Response
		if err != nil {
			resp = &Response{
				Success: false,
				Error:   fmt.Sprintf("bad request payload: %v", err),
			}
		} else {
			resp = s.dispatch(req)
		}

		raw, err := s.codec.EncodeResponse(resp)
		if err != nil {
			continue
		}

		if err := writeFrame(conn, raw); err != nil {
			return
		}
	}
}

func (s *Server) dispatch(req *Request) *Response {
	switch req.Type {
	case MsgAdd:
		if req.Add == nil {
			return &Response{Success: false, Error: "missing add payload"}
		}
		if err := s.db.Add(req.Add.Hash, req.Add.Vector); err != nil {
			return &Response{Success: false, Error: err.Error()}
		}
		return &Response{Success: true}

	case MsgGet:
		if req.Get == nil {
			return &Response{Success: false, Error: "missing get payload"}
		}
		vec, err := s.db.Get(req.Get.Hash)
		if err != nil {
			return &Response{Success: false, Error: err.Error()}
		}
		return &Response{Success: true, Vector: vec}

	case MsgRemove:
		if req.Remove == nil {
			return &Response{Success: false, Error: "missing remove payload"}
		}
		if err := s.db.Remove(req.Remove.Hash); err != nil {
			return &Response{Success: false, Error: err.Error()}
		}
		return &Response{Success: true}

	case MsgQuery:
		if req.Query == nil {
			return &Response{Success: false, Error: "missing query payload"}
		}
		results, err := s.db.Query(req.Query.Vector, req.Query.Metric, req.Query.TopK)
		if err != nil {
			return &Response{Success: false, Error: err.Error()}
		}
		return &Response{Success: true, Results: results}

	case MsgCount:
		return &Response{Success: true, Count: s.db.Count()}

	case MsgList:
		return &Response{Success: true, Vectors: s.db.List()}

	default:
		return &Response{Success: false, Error: fmt.Sprintf("unrecognized msg type: %s", req.Type)}
	}
}
