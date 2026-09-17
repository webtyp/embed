---
PLAN: "feat: webtyp/embed — el puerto Embedder, sin adaptador"
TAG: v0.1.0
EXECUTOR: unassigned
REVIEWER: none
REPO: webtyp/embed
---

> Parte del esfuerzo de búsqueda semántica nativa en el navegador. Índice maestro:
> https://github.com/webtyp/agent/blob/main/docs/PLAN.md — decisión **D4** (el puerto se
> publica en fase 2 porque `vectordb` lo importa; el adaptador es fase 3).
>
> **Alcance de este plan: solo el puerto.** El pipeline real —tokenizador, pesos, modelo
> concreto— es trabajo de fase 3 y vive en una versión reescrita de
> [`agent/docs/plans/embed.md`](https://github.com/webtyp/agent/blob/main/docs/plans/embed.md),
> que hoy describe únicamente el adaptador estático y antecede a las decisiones D4b/D5
> actuales del índice maestro (candidatos de modelo en
> [`SMALL_MODEL_FOR_EMBEDING.md`](https://github.com/webtyp/agent/blob/main/docs/SMALL_MODEL_FOR_EMBEDING.md)).
> No la uses como referencia de alcance para este plan — se reescribe antes de que la fase 3
> la necesite.
>
> **Nota de idioma:** la prosa va en español; los bloques de código mantienen sus
> comentarios en inglés, como el resto del código fuente de este repositorio.

# Plan — `webtyp/embed`, solo el puerto

## Por qué existe este repositorio antes que su implementación

`webtyp/vectordb` importa `embed.Embedder` como tipo de su `Config.Embedder` — es una
interfaz en la firma pública de otro repositorio, así que tiene que existir en alguna parte
antes de que `vectordb` compile. Ese es el motivo entero de este plan: desbloquear
`vectordb`, no entregar embeddings todavía.

**Cero ML, cero tokenizador, cero pesos, cero dependencias más allá de**
`webtyp.com/context`. Ese presupuesto es intencional: un puerto es un contrato, no una
implementación, y cargarlo con dependencias de un adaptador que ni siquiera se eligió
todavía sería exactamente el error que D4 evita en `vectordb` (importar el adaptador en vez
del puerto).

## El puerto

```go
package embed

import "webtyp.com/context"

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
```

`ID()` no es decoración. Mezclar vectores de dos modelos produce resultados que parecen
plausibles y no significan nada — la falla no tiene síntoma. Guardar el id es lo que
convierte eso en un error de arranque, en `vectordb`, no acá.

## `MockEmbedder` — obligatorio, y por qué vive acá y no en cada consumidor

```go
// MockEmbedder returns deterministic vectors derived from a hash of each text, so tests
// of vectordb (and, más adelante, de agentmemory) never depend on un modelo real.
// Vectors are L2-normalised, como exige el contrato de Embed.
type MockEmbedder struct {
	dim int
	id  string // defaults to "mock/<dim>" if empty
}

func NewMockEmbedder(dim int) *MockEmbedder
func (m *MockEmbedder) Dim() int
func (m *MockEmbedder) ID() string
func (m *MockEmbedder) Embed(ctx *context.Context, texts []string, dst []float32) error
func (m *MockEmbedder) Close() error
```

Si cada consumidor de `Embedder` escribiera su propio mock, cada uno inventaría su propio
hash y su propio criterio de determinismo — dos formas de hacer lo mismo, y una prohibida
por el skill. El mock es parte del puerto, no un detalle de test de quien lo consume.

Determinismo: el mismo texto produce siempre el mismo vector, en la misma corrida y entre
corridas (sin `time.Now`, sin `math/rand` sin semilla fija). Usá un hash estable — FNV-1a
sobre los bytes del texto alcanza — para derivar cada componente antes de normalizar.

## Tests

| Test | Verifica |
|---|---|
| `TestMockEmbedder_Deterministic` | el mismo texto produce el mismo vector en dos llamadas |
| `TestMockEmbedder_DifferentTextsDifferentVectors` | dos textos distintos no colisionan (al menos en una muestra razonable) |
| `TestMockEmbedder_DimMatchesOutput` | `len(dst) == len(texts)*Dim()` se respeta, error si no |
| `TestMockEmbedder_IsNormalised` | cada vector escrito tiene norma L2 ≈ 1 dentro de 1e-6 |
| `TestMockEmbedder_EmptyTexts` | `texts` vacío no escribe nada y no da error |

## Checklist de aceptación

```bash
go vet ./...
gotest
GOOS=js GOARCH=wasm go build ./...
grep -rn "syscall/js\|webtyp.com/storage\|webtyp.com/indexdb\|webtyp.com/tokenizer\|webtyp.com/weights" .   # → vacío
```

Después liberar, porque `vectordb` depende de este tag:

```bash
gopush 'feat: Embedder port and MockEmbedder'
```
