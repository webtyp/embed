# Agent Guide — `webtyp/embed`

Constraints for agents working on this library. **Read this before any change.**
The master index is
[`retrieval/docs/SEMANTIC_SEARCH_MASTER_PLAN.md`](https://github.com/webtyp/retrieval/blob/main/docs/SEMANTIC_SEARCH_MASTER_PLAN.md).

## What this library is

The `Embedder` contract (`Dim`, `ID`, `Embed`, `CountTokens`, `Close`) and a deterministic
`MockEmbedder`. **Nothing else.** A real model is an implementation in its own repository
(`webtyp/bekko` for `bekko-embedding-v1-a8m`), so that `vectordb`, `agentmemory` and
`retrieval`, which only need the contract, never pull a tokenizer, weights or an encoder into
their binaries.

## Dependencies — exactly these

| Import | Why |
|---|---|
| `webtyp.com/context` | every webtyp API takes it |
| `webtyp.com/fmt` | isomorphic fmt/errors |

`math` and `hash/fnv` (stdlib) are used by the mock and are cheap under TinyGo.

## The builds that define "done"

```bash
go vet ./...
gotest
gotest -tinygo
GOOS=js GOARCH=wasm go build ./...
```

## Never import these

| Never | Use instead | Why |
|---|---|---|
| `strings`, `fmt`, `errors`, `strconv` | `webtyp.com/fmt` | isomorphism + TinyGo size |
| `context` (stdlib) | `webtyp.com/context` | |
| `encoding/json` | nothing | reflection JSON costs ~1 MB of wasm |
| `map[K]V` | a slice | TinyGo's map runtime is a size tax |
| any model code (`tokenizer`, `weights`, `encoder`) | a new implementation repository | this is a contract |

## Common mistakes to avoid

- Adding a method to `Embedder` without updating every implementation (`MockEmbedder` here,
  `bekko.Embedder`) and the test doubles of its consumers (`vectordb`). It is a breaking change.
- Putting an adapter back in this repository "for convenience".
