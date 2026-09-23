---
PLAN: "feat: embed — StaticEmbedder, el adaptador real para bekko-embedding-v1-a8m"
TAG: v0.2.0
EXECUTOR: jules
REVIEWER: none
---

> This plan is dispatched via the CodeJob workflow. See skill: agents-workflow.
> Índice maestro: https://github.com/webtyp/agent/blob/main/docs/MASTER_PLAN.md — D0
> (64 dims, Matryoshka), D4/D4c, D5, fase 3.
>
> **Nota de idioma:** la prosa va en español; los bloques de código mantienen sus comentarios
> en inglés.

# Plan — `webtyp/embed`, el adaptador estático (fase 3)

## Qué existe y qué falta

`embed.go` ya tiene el puerto (`Embedder` interface) y `MockEmbedder` — no se tocan. Falta el
**adaptador real**: `StaticEmbedder`, que compone `webtyp.com/tokenizer` +
`webtyp.com/weights` + `webtyp.com/transformer` para `bekko-embedding-v1-a8m`, ya verificado
pieza por pieza en las olas anteriores (tokenizer: `Scheme` plegable, esquema `Metaspace`;
weights: formato `WTYPW1`, cuantización int8; transformer: grafo ModernBERT, `Config.Pooling`).
Esta es la primera vez que las tres piezas corren juntas.

**Ya probado en esta máquina, fuera del repo, antes de escribir este plan:** `weightsc`
corrido contra el `model.safetensors`/`config.json`/`tokenizer.json` **reales** de
`hotchpotch/bekko-embedding-v1-a8m` produce un artifact `.wtypw` de 109 MB + un `.merges` de
5,8 MB, y `weights.Open` sobre ese artifact lee correctamente sus 26 tensores (formas y
dtypes exactos, `embeddings.tok_embeddings.weight` int8 `[256000, 384]` con 256 000 escalas,
`final_norm.weight` float32 sin escalas). La cadena de producción del artifact **funciona de
punta a punta con datos reales** — lo que falta es el consumidor.

## Design gate

**1. Prior art.** El propio índice maestro ya nombra esto "adaptador estático" (D4, §5) —
el nombre `StaticEmbedder` no es una elección nueva, es la que el plan ya fijó. Patrón de
composición: igual que `agentmemory.Store` compone `orm`+`ddl`+`vectordb` sin reimplementar
ninguno, `StaticEmbedder` compone `tokenizer`+`weights`+`transformer` sin reimplementar
ninguno.

**2. Novice-name test.** `embed.NewStaticEmbedder(cfg) (Embedder, error)` — mismo patrón
`New(cfg) (T, error)` que cada repo de esta ola ya usa.

**3. Complexity ledger.**
```
Conceptos nuevos                 +1 (StaticEmbedder — pero es la ranura que D4/§5 ya reservaba)
Formas de producir un Embedder    2 (MockEmbedder para tests, StaticEmbedder para producción)
Repos que StaticEmbedder toca     0 nuevos (tokenizer/weights/transformer ya existen y están
                                   verificados; esto solo los compone)
```

**4. Dónde vive.** Acá — es la fase 3 que el índice maestro reserva para este repo
exactamente.

**5. Qué borra.** Nada existente. Cierra la fase 3.

## D0 cambió: 64 dims, no 384 — y por qué es trabajo de este adaptador

`MASTER_PLAN.md` D0 (revisado 2026-09-23): el vector que sale de `Embedder.Embed` tiene que
ser de **64 componentes**, no los 384 nativos del modelo. `bekko-embedding-v1-a8m` entrena con
Matryoshka (MRL): los primeros *k* componentes del vector de 384 ya son, por construcción, un
embedding válido para *k* ∈ {256, 128, 64} — no es una aproximación, es el modo de uso
publicado del modelo. **La verdad, `transformer.Encode` sigue devolviendo 384 — el truncado a
64 + la renormalización L2 pasan acá, en `StaticEmbedder.Embed`, antes de que el vector
llegue al llamador.** Ni `vectordb` ni `storage` se enteran de que alguna vez fueron 384.

## Cambio 1 — `dequant.go`: leer tensores cuantizados del artifact

`webtyp.com/weights` no trae un helper de dequantización — expone `Tensor.Row(i) []byte`
(bytes crudos de la fila *i*) y `Tensor.Scales []float32` (una escala por fila), y deja la
aritmética al consumidor. Es la inversa exacta de `weightsc.QuantizeRowInt8`:

```go
package embed

import "webtyp.com/weights"

// dequantRow converts one int8 row (raw bytes, one signed byte per element) back to
// float32 using its per-row scale — the exact inverse of weightsc.QuantizeRowInt8.
func dequantRow(t weights.Tensor, row int, dst []float32) {
	raw := t.Row(row)
	scale := t.Scales[row]
	for i, b := range raw {
		dst[i] = float32(int8(b)) * scale
	}
}

// dequantFull dequantizes an entire int8 tensor at once — only for tensors small enough to
// hold fully in memory (layer weights, ~31 MB total across all 4 layers of a8m — see
// Cambio 2). NEVER call this on the token embedding table (256 000 × 384, ~98 MB of int8,
// ~393 MB dequantized) — that one is read one row at a time, lazily, per token actually
// used in a batch (Cambio 4). This split is not an optimization to consider later: eagerly
// materializing the full embedding table is the difference between this running in a
// browser tab and not.
func dequantFull(t weights.Tensor) ([]float32, error) {
	if t.DType == weights.Float32 {
		return t.Float32s()
	}
	rows := t.Shape[0]
	cols := len(t.Data) / rows
	out := make([]float32, rows*cols)
	for r := 0; r < rows; r++ {
		dequantRow(t, r, out[r*cols:(r+1)*cols])
	}
	return out, nil
}
```

## Cambio 2 — `weights_bekko.go`: `transformer.Config`/`Weights` desde el artifact

Valores reales de `bekko-embedding-v1-a8m`, verificados contra su `config.json` y el header
real de `model.safetensors` (`transformer/docs/LAST_PLAN_EXECUTED.md`, etapa 3 — no los
re-derives, son estos exactos):

```go
package embed

import (
	"webtyp.com/fmt"
	"webtyp.com/transformer"
	"webtyp.com/weights"
)

func bekkoA8mConfig() transformer.Config {
	return transformer.Config{
		NumLayers:          4,
		Heads:              6,
		Dim:                384, // NATIVE — no confundir con los 64 de salida (D0)
		FFNDim:             1152,
		GlobalEveryNLayers: 3,
		LocalWindow:        128,
		GlobalRopeTheta:    160000.0,
		LocalRopeTheta:     160000.0,
		Eps:                1e-5,
		Pooling:            transformer.PoolingMean,
	}
}

// loadWeights dequantizes every layer tensor EXCEPT the token embedding table (kept lazy —
// see dequantFull's doc comment) into a transformer.Weights ready for Encode. artifact.Tensor
// panics on no code path here — every name below is checked, a missing one is a hard error,
// never a zero-valued layer silently fed into Encode.
func loadWeights(art *weights.Artifact, cfg transformer.Config) (transformer.Weights, error) {
	get := func(name string) (weights.Tensor, error) {
		t, ok := art.Tensor(name)
		if !ok {
			return weights.Tensor{}, fmt.Err("embed: artifact missing tensor ", name)
		}
		return t, nil
	}

	embedNorm, err := get("embeddings.norm.weight")
	if err != nil {
		return transformer.Weights{}, err
	}
	embedNormF, err := dequantFull(embedNorm)
	if err != nil {
		return transformer.Weights{}, err
	}

	finalNorm, err := get("final_norm.weight")
	if err != nil {
		return transformer.Weights{}, err
	}
	finalNormF, err := dequantFull(finalNorm)
	if err != nil {
		return transformer.Weights{}, err
	}

	w := transformer.Weights{EmbedNormGamma: embedNormF, FinalNormGamma: finalNormF}
	for i := 0; i < cfg.NumLayers; i++ {
		lw, err := loadLayer(get, i)
		if err != nil {
			return transformer.Weights{}, err
		}
		w.Layers = append(w.Layers, lw)
	}
	return w, nil
}
```

**Escribí `loadLayer(get func(string) (weights.Tensor, error), i int) (transformer.LayerWeights,
error)`** — dequantiza, por capa `i`: `layers.{i}.attn.Wqkv.weight` → `WqkvT`,
`layers.{i}.attn.Wo.weight` → `WoT`, `layers.{i}.mlp.Wi.weight` → `WiT`,
`layers.{i}.mlp.Wo.weight` → `MlpWoT`, `layers.{i}.mlp_norm.weight` → `MlpNormGamma` (todos
con `dequantFull`), y `layers.{i}.attn_norm.weight` → `AttnNormGamma` **solo si `i != 0`**
(la capa 0 no tiene ese tensor — es `Identity`, `AttnNormGamma` se queda `nil`; confirmalo
con `art.Tensor(...)`'s segundo valor de retorno, no asumas que falta solo en capa 0 sin
chequear).

**No asumas que `WqkvT`/`WoT`/etc. necesitan una transposición extra.** `weightsc` copia la
forma del tensor tal cual viene en `safetensors` (`[out, in]`, la convención de PyTorch) sin
transponer nada — si `transformer.MatmulT` espera exactamente esa forma (su nombre sugiere
que sí, "T" de "ya transpuesto"), cargar directo funciona. El test de Cambio 5 (coseno contra
el modelo real) es el árbitro: si el coseno no da ~1.0, **la primera sospecha es una
transposición perdida acá**, no un bug en `transformer` (que ya está verificado contra su
propio benchmark).

## Cambio 3 — `tokenizer.go` (del lado de `embed`, no confundir con el repo `tokenizer`): armar el `BPE`

```go
package embed

import (
	"webtyp.com/tokenizer"
	"webtyp.com/weights"
)

// loadTokenizer builds a ready tokenizer.BPE from the artifact's vocab and a separately
// loaded .merges file (weightsc writes them as two files on purpose — see
// weightsc/docs/LAST_PLAN_EXECUTED.md "no van los dos en el artifact"). bosID/eosID are
// bekko-embedding-v1-a8m's real special token ids from its config.json — 2 and 1
// respectively, NOT granite's 179934/179938 (tokenizer/docs/LAST_PLAN_EXECUTED.md already
// warns about this exact mistake).
func loadTokenizer(art *weights.Artifact, mergesBytes []byte) (*tokenizer.BPE, error) {
	merges := splitLines(mergesBytes) // one rule per line, same format weightsc writes
	return tokenizer.New(tokenizer.Config{
		Vocab:      art.Tokenizer.Vocab,
		Merges:     merges,
		Scheme:     tokenizer.MetaspaceScheme{},
		BosTokenID: 2,
		EosTokenID: 1,
		PadTokenID: 0,
	})
}
```

**Escribí `splitLines(b []byte) []string`** sin `strings.Split` (banned, ver `AGENTS.md`) —
un scan lineal buscando `\n`, recortando `\r` final de cada línea si aparece. `webtyp.com/fmt`
no tiene un split genérico; es más simple escribir las ~10 líneas a mano que buscar un
sustituto.

## Cambio 4 — `static_embedder.go`: `StaticEmbedder`, el `Embedder` real

```go
package embed

import (
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
		l2Normalize(out)
	}
	return nil
}

// l2Normalize scales v in place to unit length. A Matryoshka truncation is only a valid
// embedding AFTER renormalizing — the first 64 components of a unit 384-vector do not
// themselves have unit norm.
func l2Normalize(v []float32) {
	var sumSq float32
	for _, x := range v {
		sumSq += x * x
	}
	if sumSq == 0 {
		return
	}
	inv := 1.0 / sqrt32(sumSq)
	for i := range v {
		v[i] *= inv
	}
}
```

**`sqrt32(float32) float32`** — verificá si `webtyp.com/fmt` o algún paquete ya permitido
expone una raíz cuadrada para `float32` sin traer `math` completo (TinyGo soporta `math.Sqrt`
razonablemente bien en la práctica — `AGENTS.md` de `transformer` ya lo usa vía
`math.Sqrt(float64(headDim))` en `RoPE`/atención — así que `float32(math.Sqrt(float64(sumSq)))`
es aceptable acá, consistente con lo que `transformer` ya hace; no hace falta una
reimplementación a mano).

**`var _ Embedder = (*StaticEmbedder)(nil)`** al final del archivo — confirmá en compile-time
que la firma completa quedó satisfecha.

## Cambio 5 — La verificación real: coseno contra el modelo real, no contra vos mismo

**No declares esto terminado con un fixture inventado.** Igual que `transformer` etapa 2/3 y
`tokenizer` — generá 3-5 vectores de referencia corriendo el modelo real
(`AutoModel.from_pretrained("hotchpotch/bekko-embedding-v1-a8m")`, `transformers` + `torch`
CPU, mean pooling sobre `last_hidden_state` con la máscara de atención, **sin** normalizar
L2, y **sin** truncar a 64 — el fixture es el vector nativo de 384, la comparación con el
truncado se hace tomando los primeros 64 de AMBOS lados antes de normalizar) para 3-5
oraciones en español. Embebé solo los vectores en `testdata/reference_vectors.json` (mismo
patrón que `transformer` — el artifact de 109 MB no se sube al repo).

`TestStaticEmbedder_MatchesReference`: para cada oración, `Embed` con `StaticEmbedder` real
(cargando el artifact real — sí hace falta el archivo de 109 MB para correr este test
puntual; documentá en el test cómo generarlo: `weightsc -in <dir-con-los-3-archivos-reales>
-out ... -merges-out ... -id ... -version 1`, dónde bajar los 3 archivos de entrada, y que el
test se salta con `t.Skip` si el artifact no está presente en `testdata/` — no lo comitees,
109 MB es demasiado para este repo) y verificá coseno ≥ 0.999 contra la referencia (mismo
truncado a 64 + renormalización de los dos lados antes de comparar).

**Si el coseno no da ~1.0:** la lista de sospechosos, en orden de probabilidad — (1) una
transposición de `Wqkv`/`Wo`/`Wi` perdida en `loadLayer` (Cambio 2, ya avisado), (2) el orden
de `attn_norm` ausente en capa 0 mal indexado, (3) el tokenizer produciendo ids distintos a
los reales (poco probable, `tokenizer` ya tiene su propio fixture verificado, pero confirmalo
tokenizando la misma oración con `AutoTokenizer` real y comparando ids antes de sospechar de
`transformer`).

## Tests

| Test | Verifica |
|---|---|
| `TestStaticEmbedder_MatchesReference` | Cambio 5 — la puerta real |
| `TestL2Normalize` | un vector no-unitario queda con norma 1; el vector cero queda sin cambios (no `NaN`) |
| `TestDequantRow_RoundTrip` | cuantizar con `weightsc.QuantizeRowInt8` y dequantizar acá vuelve dentro de `scale/2` del original (mismo criterio que el test homónimo de `weightsc`) |
| `TestEmbed_DstLengthMismatch` | `len(dst) != len(texts)*64` devuelve error, no panic |
| `TestEmbed_EmptyTexts` | `texts` vacío no falla, no escribe nada |
| `TestStaticEmbedder_Dim` | `Dim() == 64`, no 384 |

## Checklist de aceptación

```bash
go vet ./...
gotest
gotest -tinygo
GOOS=js GOARCH=wasm go build ./...
tinygo build -target wasm -o /dev/null .
grep -rn "map\[" --include="*.go" . | grep -v _test.go                     # → vacío
grep -rn '"strings"\|"context"\|"encoding/json"' --include="*.go" . | grep -v _test.go  # → vacío
```

`tinygo build -target wasm -o /dev/null .` **puede fallar por "no es un paquete main"**
(mismo caso benigno que `transformer`/`tokenizer` — no tienen `func main`) — si falla con ese
mensaje exacto, no es un bug; `gotest -tinygo` es la puerta real acá también.
