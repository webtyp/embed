package embed_test

import (
	"math"
	"testing"

	"webtyp.com/context"
	"webtyp.com/embed"
)

func TestMockEmbedder_Deterministic(t *testing.T) {
	ctx := context.Background()
	m := embed.NewMockEmbedder(128)

	texts := []string{"hello world", "webtyp semantic search"}
	dst1 := make([]float32, len(texts)*m.Dim())
	dst2 := make([]float32, len(texts)*m.Dim())

	if err := m.Embed(ctx, texts, dst1); err != nil {
		t.Fatalf("Embed failed: %v", err)
	}
	if err := m.Embed(ctx, texts, dst2); err != nil {
		t.Fatalf("Embed failed: %v", err)
	}

	for i := range dst1 {
		if dst1[i] != dst2[i] {
			t.Errorf("mismatch at index %d: %f != %f", i, dst1[i], dst2[i])
		}
	}
}

func TestMockEmbedder_DifferentTextsDifferentVectors(t *testing.T) {
	ctx := context.Background()
	m := embed.NewMockEmbedder(64)

	texts1 := []string{"text sample A"}
	texts2 := []string{"text sample B"}

	dst1 := make([]float32, m.Dim())
	dst2 := make([]float32, m.Dim())

	if err := m.Embed(ctx, texts1, dst1); err != nil {
		t.Fatalf("Embed failed: %v", err)
	}
	if err := m.Embed(ctx, texts2, dst2); err != nil {
		t.Fatalf("Embed failed: %v", err)
	}

	identical := true
	for i := range dst1 {
		if dst1[i] != dst2[i] {
			identical = false
			break
		}
	}
	if identical {
		t.Errorf("different texts produced identical vectors")
	}
}

func TestMockEmbedder_DimMatchesOutput(t *testing.T) {
	ctx := context.Background()
	m := embed.NewMockEmbedder(32)

	texts := []string{"foo", "bar"}
	validDst := make([]float32, len(texts)*m.Dim())

	if err := m.Embed(ctx, texts, validDst); err != nil {
		t.Errorf("expected success with correct dst length, got: %v", err)
	}

	invalidDst := make([]float32, len(texts)*m.Dim()-1)
	if err := m.Embed(ctx, texts, invalidDst); err == nil {
		t.Errorf("expected error with invalid dst length, got nil")
	}
}

func TestMockEmbedder_IsNormalised(t *testing.T) {
	ctx := context.Background()
	dim := 128
	m := embed.NewMockEmbedder(dim)

	texts := []string{"apple", "banana", "cherry"}
	dst := make([]float32, len(texts)*dim)

	if err := m.Embed(ctx, texts, dst); err != nil {
		t.Fatalf("Embed failed: %v", err)
	}

	for i := range texts {
		vec := dst[i*dim : (i+1)*dim]
		var normSq float64
		for _, v := range vec {
			normSq += float64(v) * float64(v)
		}
		norm := math.Sqrt(normSq)
		if math.Abs(norm-1.0) > 1e-6 {
			t.Errorf("text %q vector L2 norm = %f; want ~1.0 within 1e-6", texts[i], norm)
		}
	}
}

func TestMockEmbedder_EmptyTexts(t *testing.T) {
	ctx := context.Background()
	m := embed.NewMockEmbedder(64)

	var texts []string
	var dst []float32

	if err := m.Embed(ctx, texts, dst); err != nil {
		t.Fatalf("expected no error on empty texts, got: %v", err)
	}
}

func TestMockEmbedder_IDAndDimAndClose(t *testing.T) {
	m := embed.NewMockEmbedder(256)
	if m.Dim() != 256 {
		t.Errorf("Dim() = %d, want 256", m.Dim())
	}
	if m.ID() != "mock/256" {
		t.Errorf("ID() = %q, want %q", m.ID(), "mock/256")
	}
	if err := m.Close(); err != nil {
		t.Errorf("Close() error = %v, want nil", err)
	}
}
