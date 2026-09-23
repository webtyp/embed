package embed

import (
	"webtyp.com/tokenizer"
	"webtyp.com/weights"
)

// splitLines splits a byte slice by line breaks without using stdlib strings or bytes package.
func splitLines(b []byte) []string {
	if len(b) == 0 {
		return nil
	}
	var lines []string
	start := 0
	for i := 0; i < len(b); i++ {
		if b[i] == '\n' {
			line := b[start:i]
			if len(line) > 0 && line[len(line)-1] == '\r' {
				line = line[:len(line)-1]
			}
			lines = append(lines, string(line))
			start = i + 1
		}
	}
	if start < len(b) {
		line := b[start:]
		if len(line) > 0 && line[len(line)-1] == '\r' {
			line = line[:len(line)-1]
		}
		lines = append(lines, string(line))
	}
	return lines
}

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
