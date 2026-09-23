package embed_test

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"webtyp.com/context"
	"webtyp.com/embed"
)

// refCase mirrors testdata/reference_vectors.json — real 384-dim vectors from
// AutoModel.from_pretrained("hotchpotch/bekko-embedding-v1-a8m") (transformers + torch CPU,
// eager attention, mean pooling over last_hidden_state with the attention mask, NOT
// L2-normalized, NOT truncated). encoding/json is fine in this file: it is host-only test
// code (AGENTS.md's stdlib table bans it from production code, not from *_test.go loading
// fixtures that never ship to wasm).
type refCase struct {
	Text      string    `json:"text"`
	Vector384 []float32 `json:"vector384"`
}

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

// TestStaticEmbedder_MatchesReference is the real gate (docs/PLAN.md Cambio 5): does the
// composed tokenizer+weights+transformer pipeline actually reproduce the real model's
// output, not just "run without error." testdata/reference_vectors.json ships in the repo
// (three sentences, real 384-dim vectors — small). The 109 MB artifact/.merges do NOT — this
// test skips without them; generate locally with weightsc against the real
// hotchpotch/bekko-embedding-v1-a8m files (model.safetensors, config.json, tokenizer.json)
// and drop the output at the two paths below.
func TestStaticEmbedder_MatchesReference(t *testing.T) {
	artBytes, err := os.ReadFile("testdata/bekko-embedding-v1-a8m.wtypw")
	if err != nil {
		t.Skip("skipping reference test: testdata/bekko-embedding-v1-a8m.wtypw not found")
	}
	mergesBytes, err := os.ReadFile("testdata/bekko-embedding-v1-a8m.merges")
	if err != nil {
		t.Skip("skipping reference test: testdata/bekko-embedding-v1-a8m.merges not found")
	}
	refJSON, err := os.ReadFile("testdata/reference_vectors.json")
	if err != nil {
		t.Fatalf("testdata/reference_vectors.json missing (should be committed): %v", err)
	}
	var refs []refCase
	if err := json.Unmarshal(refJSON, &refs); err != nil {
		t.Fatalf("parsing reference_vectors.json: %v", err)
	}

	emb, err := embed.NewStaticEmbedder(embed.Config{
		ArtifactBytes: artBytes,
		MergesBytes:   mergesBytes,
	})
	if err != nil {
		t.Fatalf("failed to create StaticEmbedder: %v", err)
	}

	if emb.Dim() != 128 {
		t.Fatalf("Dim() = %d; want 128", emb.Dim())
	}

	ctx := context.Background()
	for _, ref := range refs {
		t.Run(ref.Text, func(t *testing.T) {
			if len(ref.Vector384) != 384 {
				t.Fatalf("reference vector for %q has %d components, want 384", ref.Text, len(ref.Vector384))
			}

			got := make([]float32, emb.Dim())
			if err := emb.Embed(ctx, []string{ref.Text}, got); err != nil {
				t.Fatalf("Embed failed: %v", err)
			}

			var normSq float64
			for _, v := range got {
				normSq += float64(v) * float64(v)
			}
			if math.Abs(math.Sqrt(normSq)-1.0) > 1e-5 {
				t.Fatalf("vector L2 norm = %f, want ~1.0", math.Sqrt(normSq))
			}

			// Truncate the REAL model's native 384-dim output to the same 128 and
			// renormalize both sides identically before comparing — this is the same
			// Matryoshka rule Embed itself applies (docs/PLAN.md D0), applied here to the
			// ground truth so the comparison is apples-to-apples.
			want := append([]float32(nil), ref.Vector384[:emb.Dim()]...)
			embed.L2Normalize(want)

			cos := cosineSimilarity(got, want)
			if cos < 0.999 {
				t.Errorf("cosine(StaticEmbedder, real model) = %v, want >= 0.999 (text: %q)", cos, ref.Text)
			}
		})
	}
}

func cosineSimilarity(a, b []float32) float64 {
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}
