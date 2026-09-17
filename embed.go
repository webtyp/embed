package embed

import (
	"fmt"
	"hash/fnv"
	"math"

	"webtyp.com/context"
)

// Embedder turns text into vectors. Implementations run entirely in the browser.
type Embedder interface {
	// Dim is the vector dimension. Constant for the lifetime of the Embedder.
	Dim() int

	// ID identifies the model that produced these vectors, e.g.
	// "granite-embedding-97m-multilingual-r2/int8". Vectors from different models are NOT
	// comparable; vectordb stores this and refuses a corpus that disagrees.
	ID() string

	// Embed writes one vector per text into dst, which MUST have length
	// len(texts)*Dim(). Vectors are L2-normalised on the way out (master index D1).
	// The caller owns dst, so batching allocates once.
	Embed(ctx *context.Context, texts []string, dst []float32) error

	Close() error
}

// MockEmbedder returns deterministic vectors derived from a hash of each text, so tests
// of vectordb (and agentmemory) never depend on a real model.
// Vectors are L2-normalised, as required by the Embed contract.
type MockEmbedder struct {
	dim int
	id  string // defaults to "mock/<dim>" if empty
}

// NewMockEmbedder creates a MockEmbedder with the given vector dimension.
func NewMockEmbedder(dim int) *MockEmbedder {
	return &MockEmbedder{
		dim: dim,
		id:  fmt.Sprintf("mock/%d", dim),
	}
}

// Dim returns the vector dimension.
func (m *MockEmbedder) Dim() int {
	return m.dim
}

// ID returns the identifier of the mock embedder.
func (m *MockEmbedder) ID() string {
	if m.id == "" {
		return fmt.Sprintf("mock/%d", m.dim)
	}
	return m.id
}

// Embed writes deterministic L2-normalised vectors into dst.
func (m *MockEmbedder) Embed(ctx *context.Context, texts []string, dst []float32) error {
	expectedLen := len(texts) * m.dim
	if len(dst) != expectedLen {
		return fmt.Errorf("dst length mismatch: expected %d (len(texts)*Dim()), got %d", expectedLen, len(dst))
	}
	if len(texts) == 0 {
		return nil
	}

	for i, text := range texts {
		offset := i * m.dim
		vec := dst[offset : offset+m.dim]

		// Hash text and component index to generate deterministic non-zero values
		var normSq float64
		for j := 0; j < m.dim; j++ {
			h := fnv.New64a()
			h.Write([]byte(text))
			var b [4]byte
			b[0] = byte(j)
			b[1] = byte(j >> 8)
			b[2] = byte(j >> 16)
			b[3] = byte(j >> 24)
			h.Write(b[:])
			val := float64(int64(h.Sum64())) / float64(math.MaxInt64)
			if val == 0 {
				val = 0.1
			}
			vec[j] = float32(val)
			normSq += val * val
		}

		norm := math.Sqrt(normSq)
		if norm > 0 {
			for j := 0; j < m.dim; j++ {
				vec[j] = float32(float64(vec[j]) / norm)
			}
		}
	}

	return nil
}

// Close is a no-op for MockEmbedder.
func (m *MockEmbedder) Close() error {
	return nil
}

// Ensure MockEmbedder implements Embedder.
var _ Embedder = (*MockEmbedder)(nil)
