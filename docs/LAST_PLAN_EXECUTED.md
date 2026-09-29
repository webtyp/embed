---
PLAN: "refactor!: embed is the contract only — StaticEmbedder moved to webtyp/bekko"
TAG: v0.4.0
EXECUTOR: jules
REVIEWER: none
---

> This plan is dispatched via the CodeJob workflow. See skill: agents-workflow.
>
> Part of
> [`AGENT_ECOSYSTEM_MASTER_PLAN.md`](https://github.com/webtyp/agent/blob/main/docs/AGENT_ECOSYSTEM_MASTER_PLAN.md).
> **Blocked until `webtyp.com/bekko` v0.1.0 is published.**

# Plan — `webtyp/embed` keeps only the contract

## 0. Context

`embed` held the `Embedder` contract **and** its real implementation, `StaticEmbedder`. The
implementation now lives in `webtyp.com/bekko` (as `bekko.Embedder`), published and verified.
This plan deletes the copy here, so that importing the contract no longer pulls a tokenizer,
a weight reader and an encoder into a browser binary.

Nobody outside this repository references `StaticEmbedder`, `NewStaticEmbedder`, `Config` or
`L2Normalize` (checked across `vectordb`, `agentmemory`, `agent`, `retrieval`).

## Development rules (inline)

- Contract library: after this plan the only non-test code is `embed.go` (interface + mock).
- **Never import:** `fmt`, `errors`, `strings`, `strconv` (use `webtyp.com/fmt`), `context`
  (use `webtyp.com/context`), `encoding/json`, `map[K]V`, `os`, `log`. `math` and `hash/fnv`
  stay, because the mock uses them today.
- Tests: `testing` only. Do **not** run `gopush`/`codejob`.

## Design gate

No API is added. **What it deletes:** `static_embedder.go`, `dequant.go`, `dequant_test.go`,
`tokenizer.go`, `weights_bekko.go`, `static_embedder_test.go`, `count_internal_test.go`,
`testdata/`, and from `go.mod` the requirements on `tokenizer`, `transformer`, `weights`
(`go mod tidy`). The exported symbols `StaticEmbedder`, `NewStaticEmbedder`, `Config`,
`L2Normalize` disappear (breaking, hence v0.4.0). Their replacement is `webtyp.com/bekko`.

## Stage 1 — delete

Delete the files listed above. Run `go mod tidy`.

## Stage 2 — docs

- `README.md` and `AGENTS.md`: describe the repository as "the `Embedder` contract and a
  deterministic `MockEmbedder`". Replace every description of `StaticEmbedder` with one
  line: "The real implementation for `bekko-embedding-v1-a8m` is `webtyp.com/bekko`."
  `AGENTS.md`'s dependency table keeps only `webtyp.com/context` and `webtyp.com/fmt`.

## Stages

| Stage | Files | Acceptance |
|---|---|---|
| 1 | deletions, `go.mod` | `ls *.go` → `embed.go embed_test.go`; `grep -n "tokenizer\|transformer\|weights" go.mod` → empty; `gotest`, `gotest -tinygo` pass |
| 2 | `README.md`, `AGENTS.md` | `grep -rn "StaticEmbedder" README.md AGENTS.md` → empty |
