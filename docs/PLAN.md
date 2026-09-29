---
PLAN: "feat!: Embedder.CountTokens — the token count of a text as the model reads it"
TAG: v0.3.0
EXECUTOR: jules
REVIEWER: none
STATUS: running
SESSION: 2020630610221412284
---

> This plan is dispatched via the CodeJob workflow. See skill: agents-workflow.
>
> Part of
> [`AGENT_ECOSYSTEM_MASTER_PLAN.md`](https://github.com/webtyp/agent/blob/main/docs/AGENT_ECOSYSTEM_MASTER_PLAN.md)
> (open decision 5, resolved). Independent of every other phase. `webtyp/retrieval`'s first
> plan (chunking) waits for this tag.

# Plan — `webtyp.com/embed`: let callers see how many tokens a text costs

## 0. Context

`embed.Embedder` turns text into vectors inside the browser. `webtyp/retrieval` is about to
split uploaded documents into chunks of **at most 256 tokens** (decision D4c of the
semantic-search master plan, in
https://github.com/webtyp/retrieval/blob/main/docs/SEMANTIC_SEARCH_MASTER_PLAN.md). A token
is the unit the model reads. It is not a word and not a character, and only the model's own
tokenizer knows how many tokens a text is.

Today nobody outside `embed` can ask that question. The tokenizer is private to
`StaticEmbedder`, and the `Embedder` interface has no method for it. The chunker would have to
guess from characters, and a guess would sometimes produce chunks longer than 256 tokens. This
plan adds the method to the contract, so every embedder answers with its own tokenizer.

## Development rules (inline)

- **Primary runtime is a browser tab compiled with TinyGo.** Every file compiles under
  `GOOS=js GOARCH=wasm` and TinyGo.
- **Never import:** `strings`, `fmt`, `errors`, `strconv` (use `webtyp.com/fmt`), `context`
  (use `webtyp.com/context`), `encoding/json`, `map[K]V`, `os`, `log`, `net/http`.
  `math` stays allowed, as today.
- Do not change tokenization, weight loading or the encoder. This plan only **exposes** a
  count that `Embed` already computes internally.
- Tests: `testing` only. Do **not** run `gopush`/`codejob`.

## Design gate (api-design — five answers)

1. **Prior art.** **OpenAI** ships `tiktoken` so callers count tokens before sending text
   to its embedding endpoint (max 8191 tokens). **Google Gemini** has a `countTokens` method
   on the same model object that embeds. **Hugging Face `transformers`**: callers run the
   model's `tokenizer(text)` and read `len(input_ids)` to chunk for sentence-transformers.
   In all three, **the model's own tokenizer answers**, next to the embedding call. Here the
   tokenizer is hidden inside the adapter, so the answer has to be a method on the contract.
2. **Novice-name test.** `embedder.CountTokens(text)` reads as "count the tokens of this
   text". It is the same name and signature as `llm.TokenCounter.CountTokens` in
   `webtyp/llm`, so a developer learns one word for one idea. Because Go interfaces are
   structural, every `Embedder` also satisfies `llm.TokenCounter` without importing it.
3. **Complexity ledger.**
   ```
   Concepts the developer must learn   +1 (CountTokens) / −0
   Files they must touch to do X       +0 / −0
   Lines at the call site              +1 / −N   (the chunker stops guessing from characters)
   Ways to do the same thing           +0 / −0
   ```
4. **Where it belongs.** The count depends on the tokenizer, and the tokenizer belongs to the
   embedding model, which is exactly what an `Embedder` wraps. Putting it anywhere else would
   need a second copy of the model's vocabulary.
5. **What it deletes.** Nothing. Adding a method to an interface is a **breaking change** for
   any type implementing `Embedder` outside this repository. The one known case is the test
   double `fixedVectorEmbedder` in `webtyp/vectordb/vectordb_test.go`, which gains a
   one-line `CountTokens` when `vectordb` bumps to v0.3.0 (tracked in the master plan, not
   in this plan).

## Stage 1 — the contract (`embed.go`)

Add to `Embedder`, after `Embed`:

```go
	// CountTokens returns how many tokens text becomes when this model reads it, including
	// any special tokens the model adds. Embed of a text reads exactly this many tokens.
	CountTokens(text string) int
```

## Stage 2 — the implementations

**`static_embedder.go`**:

```go
// CountTokens is the length of the token sequence Embed feeds the encoder for text.
func (e *StaticEmbedder) CountTokens(text string) int {
	return len(e.bpe.Encode(nil, text))
}
```

It must call the **same** `e.bpe.Encode(nil, text)` that `Embed` calls. Never approximate it.

**`embed.go`**, `MockEmbedder`:

```go
// CountTokens counts space-separated words: deterministic and dependency-free, like the
// rest of the mock. It is not any real model's count.
func (m *MockEmbedder) CountTokens(text string) int
```

Implementation: count maximal runs of bytes that are not `' '`, `'\t'`, `'\n'` or `'\r'`.
Write it by hand, because the `strings` package is not allowed.

## Stage 3 — tests

| Test | File | Asserts |
|---|---|---|
| `TestMockEmbedder_CountTokens` | `embed_test.go` | `""` → 0; `"hola"` → 1; `"  hola   mundo \n"` → 2 |
| `TestStaticEmbedder_CountTokensMatchesEmbed` | **new** `count_internal_test.go`, `package embed` (internal, because it reads the unexported `e.bpe`) | loads `testdata/bekko-embedding-v1-a8m.wtypw` and `.merges` exactly like `TestStaticEmbedder_MatchesReference` does, and calls `t.Skip` with the same messages when they are absent. For `"hola mundo"`, `"¿Cuál es la duración del contrato?"` and a 2 000-character text, `CountTokens(s) == len(e.bpe.Encode(nil, s))` and the value is > 0 |
| `TestEmbedder_SatisfiesTokenCounterShape` | `embed_test.go` | declares in the test file `type tokenCounter interface{ CountTokens(string) int }` and asserts `var _ tokenCounter = embed.Embedder(nil)`, which proves `llm.TokenCounter` is satisfied structurally without importing `llm` |

## Stage 4 — docs

In `README.md` (and `AGENTS.md` if it lists the interface), add `CountTokens` to the
description of `Embedder`, with one sentence: "use it to keep chunks under the model's token
limit".

## Stages

| Stage | Files | Acceptance |
|---|---|---|
| 1 | `embed.go` | the interface has `CountTokens(text string) int` |
| 2 | `static_embedder.go`, `embed.go` | `var _ Embedder = (*MockEmbedder)(nil)` and the StaticEmbedder assertion compile |
| 3 | `embed_test.go`, `count_internal_test.go` | `gotest` and `gotest -tinygo` pass |
| 4 | `README.md`, `AGENTS.md` | `grep -n CountTokens README.md` finds the line |
