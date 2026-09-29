package embed

import (
	"os"
	"testing"
)

func TestStaticEmbedder_CountTokensMatchesEmbed(t *testing.T) {
	artBytes, err := os.ReadFile("testdata/bekko-embedding-v1-a8m.wtypw")
	if err != nil {
		t.Skip("skipping token count test: testdata/bekko-embedding-v1-a8m.wtypw not found")
	}
	mergesBytes, err := os.ReadFile("testdata/bekko-embedding-v1-a8m.merges")
	if err != nil {
		t.Skip("skipping token count test: testdata/bekko-embedding-v1-a8m.merges not found")
	}

	emb, err := NewStaticEmbedder(Config{
		ArtifactBytes: artBytes,
		MergesBytes:   mergesBytes,
	})
	if err != nil {
		t.Fatalf("failed to create StaticEmbedder: %v", err)
	}

	// Build a 2000-character text
	longText := ""
	for len(longText) < 2000 {
		longText += "Esta es una prueba de texto largo para verificar el conteo de tokens. "
	}

	texts := []string{
		"hola mundo",
		"¿Cuál es la duración del contrato?",
		longText,
	}

	for _, s := range texts {
		cnt := emb.CountTokens(s)
		expectedLen := len(emb.bpe.Encode(nil, s))
		if cnt != expectedLen {
			t.Errorf("CountTokens(%q) = %d, want %d", s, cnt, expectedLen)
		}
		if cnt <= 0 {
			t.Errorf("CountTokens(%q) = %d, want > 0", s, cnt)
		}
	}
}
