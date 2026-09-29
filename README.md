# embed
<img src="docs/img/badges.svg">

The `Embedder` contract for browser-native semantic search: text in, L2-normalized vectors out,
plus `CountTokens` to keep chunks under the model's token limit. It also ships a deterministic
`MockEmbedder` for tests. It is a contract only, with no model code, so importing it adds nothing
heavy to a browser binary.

| I want to… | Use |
|---|---|
| depend on "something that embeds text" | `embed.Embedder` |
| test without a model | `embed.NewMockEmbedder(dim)` |
| the real `bekko-embedding-v1-a8m` model | [`webtyp.com/bekko`](https://github.com/webtyp/bekko) (`bekko.New`) |

## Documentation

- [Agent guide](AGENTS.md): rules for anyone changing this library.
- [Last executed plan](docs/LAST_PLAN_EXECUTED.md): why the real adapter moved to `webtyp/bekko`.
