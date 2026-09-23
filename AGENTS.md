# Agent Guide — `webtyp/embed`

Constraints for agents working on this library. **Read this before any change.**
The current work order is [docs/PLAN.md](docs/PLAN.md); the master index is
[`agent/docs/MASTER_PLAN.md`](https://github.com/webtyp/agent/blob/main/docs/MASTER_PLAN.md).

---

## What this library is

The `Embedder` port (text → vectors) plus its implementations. `MockEmbedder` (already
shipped) is deterministic and dependency-free, for tests. `StaticEmbedder` is the real
adapter: it composes `webtyp.com/tokenizer` + `webtyp.com/weights` + `webtyp.com/transformer`
for exactly one model, `bekko-embedding-v1-a8m` — it implements none of the three itself.

**Its primary runtime is a browser tab compiled with TinyGo.** The host (`go test`) is a
development convenience — `Embed` runs in-process inside `webtyp/vectordb`'s call path, in
the browser, once per query and once per document chunk (≤256 tokens, `MASTER_PLAN.md` D4c).

---

## Dependencies — exactly these, nothing else without a plan

| Import | Why |
|---|---|
| `webtyp.com/tokenizer` | text → token ids |
| `webtyp.com/weights` | reads the `.wtypw` artifact |
| `webtyp.com/transformer` | the encoder graph |
| `webtyp.com/context` | every webtyp API takes this |
| `webtyp.com/fmt` | isomorphic fmt/errors |

`StaticEmbedder` never imports `webtyp.com/fetch` or decides where the artifact bytes come
from — `Config.ArtifactBytes`/`MergesBytes` are injected already-fetched. Downloading and
IndexedDB caching (`MASTER_PLAN.md` D5) are the caller's job, not this adapter's.

---

## The builds that define "done"

```bash
go vet ./...
gotest
gotest -tinygo
GOOS=js GOARCH=wasm go build ./...
tinygo build -target wasm -o /dev/null .
```

The last one fails with `expected main package to have name "main"` regardless of
correctness — this is a library package with no `main`, same as `transformer` and
`tokenizer`. `gotest -tinygo` is the real compile-and-test-under-TinyGo gate.

---

## Never import these

| Never | Use instead | Why |
|---|---|---|
| `strings`, `fmt`, `errors`, `strconv` | `webtyp.com/fmt` | isomorphism + TinyGo size |
| `context` (stdlib) | `webtyp.com/context` | every webtyp API takes this one |
| `encoding/json` | hand-written parsing, or `webtyp.com/json` for `model.Encodable` types | reflection-based JSON costs ~1 MB of wasm |
| `map[K]V` | a slice scanned linearly | TinyGo's map runtime is a size tax on every binary that imports this |
| `os`, `log`, `net/http` | inject it / `webtyp.com/fetch` | a library never touches the process environment |

`math.Sqrt` (stdlib) is fine — `webtyp/transformer` already uses it for the same reason
(RoPE/attention), and it compiles cheaply under TinyGo. It is the one stdlib exception this
repo inherits from its sibling.

---

## Memory shape — this is not optional

The token embedding table (`embeddings.tok_embeddings.weight`, 256 000 × 384 int8) is
**never dequantized in full** — that would be ~393 MB of float32 for a table 99% of which is
irrelevant to any single call. Every other weight tensor (the 4 transformer layers, ~31 MB
dequantized) is small enough to dequantize once at construction and keep resident. If you
find yourself materializing the full embedding table, stop — re-read `dequant.go`'s doc
comments.

---

## Common mistakes to avoid

- Reimplementing tokenization, weight parsing, or the encoder graph "to make this file
  self-contained." All three already exist, verified, in their own repos — this repo
  composes them.
- Skipping Matryoshka truncation's renormalization step. The first 64 components of a unit
  384-vector do not themselves have unit norm — `Embed` must L2-normalize AFTER truncating,
  every time.
- Trusting a synthetic/hand-built test fixture for `TestStaticEmbedder_MatchesReference`.
  The whole point of that test is comparing against the real model's real output — see
  `docs/PLAN.md` Cambio 5.
