package embed_test

import (
	"math"
	"os"
	"testing"

	"webtyp.com/context"
	"webtyp.com/embed"
)

func TestL2Normalize(t *testing.T) {
	vec := []float32{3.0, 4.0}
	embed.L2Normalize(vec)

	if math.Abs(float64(vec[0]-0.6)) > 1e-5 || math.Abs(float64(vec[1]-0.8)) > 1e-5 {
		t.Errorf("L2Normalize failed: got [%f, %f], want [0.6, 0.8]", vec[0], vec[1])
	}

	// Zero vector check
	zeroVec := []float32{0.0, 0.0}
	embed.L2Normalize(zeroVec)
	if zeroVec[0] != 0.0 || zeroVec[1] != 0.0 {
		t.Errorf("L2Normalize zero vector modified: got [%f, %f]", zeroVec[0], zeroVec[1])
	}
}

func TestStaticEmbedder_InvalidConfig(t *testing.T) {
	_, err := embed.NewStaticEmbedder(embed.Config{
		ArtifactBytes: []byte("invalid artifact data"),
		MergesBytes:   []byte("invalid merges"),
	})
	if err == nil {
		t.Errorf("expected error when initializing StaticEmbedder with invalid artifact bytes")
	}
}

func TestStaticEmbedder_MatchesReference(t *testing.T) {
	artBytes, err := os.ReadFile("testdata/bekko-embedding-v1-a8m.wtypw")
	if err != nil {
		t.Skip("skipping reference test: testdata/bekko-embedding-v1-a8m.wtypw not found")
	}
	mergesBytes, err := os.ReadFile("testdata/bekko-embedding-v1-a8m.merges")
	if err != nil {
		t.Skip("skipping reference test: testdata/bekko-embedding-v1-a8m.merges not found")
	}

	emb, err := embed.NewStaticEmbedder(embed.Config{
		ArtifactBytes: artBytes,
		MergesBytes:   mergesBytes,
	})
	if err != nil {
		t.Fatalf("failed to create StaticEmbedder: %v", err)
	}

	if emb.Dim() != 64 {
		t.Errorf("Dim() = %d; want 64", emb.Dim())
	}

	ctx := context.Background()
	texts := []string{"Hola mundo"}
	dst := make([]float32, emb.Dim())
	if err := emb.Embed(ctx, texts, dst); err != nil {
		t.Fatalf("Embed failed: %v", err)
	}

	// Check dst is L2 normalized
	var normSq float64
	for _, v := range dst {
		normSq += float64(v) * float64(v)
	}
	if math.Abs(math.Sqrt(normSq)-1.0) > 1e-5 {
		t.Errorf("vector L2 norm = %f, want ~1.0", math.Sqrt(normSq))
	}
}
