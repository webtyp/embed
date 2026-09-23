package embed

import (
	"math"

	"webtyp.com/context"
	"webtyp.com/fmt"
	"webtyp.com/tokenizer"
	"webtyp.com/transformer"
	"webtyp.com/weights"
)

const truncatedDim = 64 // D0 — Matryoshka, no los 384 nativos

// Config assembles a StaticEmbedder. Both byte slices are already fetched — this package
// never imports webtyp.com/fetch or decides where the artifact lives (D5's IndexedDB
// caching, the download itself: the caller's job, not this adapter's).
type Config struct {
	ArtifactBytes []byte // the .wtypw file
	MergesBytes   []byte // the companion .merges file, same run of weightsc
}

type StaticEmbedder struct {
	id     string
	art    *weights.Artifact
	tokEmb weights.Tensor // token embedding table — NEVER dequantized in full, see dequantFull
	cfg    transformer.Config
	w      transformer.Weights
	bpe    *tokenizer.BPE
}

func NewStaticEmbedder(cfg Config) (*StaticEmbedder, error) {
	art, err := weights.Open(cfg.ArtifactBytes)
	if err != nil {
		return nil, fmt.Err("embed: opening artifact: ", err)
	}
	tokEmb, ok := art.Tensor("embeddings.tok_embeddings.weight")
	if !ok {
		return nil, fmt.Err("embed: artifact missing embeddings.tok_embeddings.weight")
	}

	tcfg := bekkoA8mConfig()
	w, err := loadWeights(art, tcfg)
	if err != nil {
		return nil, err
	}
	bpe, err := loadTokenizer(art, cfg.MergesBytes)
	if err != nil {
		return nil, err
	}

	return &StaticEmbedder{
		id:     fmt.Sprintf("%s/d%d", art.ID, truncatedDim),
		art:    art,
		tokEmb: tokEmb,
		cfg:    tcfg,
		w:      w,
		bpe:    bpe,
	}, nil
}

func (e *StaticEmbedder) Dim() int { return truncatedDim }

func (e *StaticEmbedder) ID() string { return e.id }

func (e *StaticEmbedder) Close() error { return nil }

func (e *StaticEmbedder) Embed(ctx *context.Context, texts []string, dst []float32) error {
	expected := len(texts) * truncatedDim
	if len(dst) != expected {
		return fmt.Err("embed: dst length mismatch, expected ", expected, " got ", len(dst))
	}
	if len(texts) == 0 {
		return nil
	}

	for i, text := range texts {
		ids := e.bpe.Encode(nil, text)
		seqLen := len(ids)
		if seqLen == 0 {
			return fmt.Err("embed: empty token sequence for text at index ", i)
		}

		tokenEmbeds := make([]float32, seqLen*e.cfg.Dim)
		for pos, id := range ids {
			if int(id) < 0 || int(id) >= e.tokEmb.Shape[0] {
				return fmt.Err("embed: token id out of vocab range: ", id)
			}
			dequantRow(e.tokEmb, int(id), tokenEmbeds[pos*e.cfg.Dim:(pos+1)*e.cfg.Dim])
		}

		pooled, err := transformer.Encode(e.cfg, e.w, tokenEmbeds, seqLen)
		if err != nil {
			return fmt.Err("embed: Encode: ", err)
		}

		out := dst[i*truncatedDim : (i+1)*truncatedDim]
		copy(out, pooled[:truncatedDim])
		L2Normalize(out)
	}
	return nil
}

// L2Normalize scales v in place to unit length. A Matryoshka truncation is only a valid
// embedding AFTER renormalizing — the first 64 components of a unit 384-vector do not
// themselves have unit norm.
func L2Normalize(v []float32) {
	var sumSq float64
	for _, x := range v {
		sumSq += float64(x) * float64(x)
	}
	if sumSq == 0 {
		return
	}
	inv := float32(1.0 / math.Sqrt(sumSq))
	for i := range v {
		v[i] *= inv
	}
}

var _ Embedder = (*StaticEmbedder)(nil)
