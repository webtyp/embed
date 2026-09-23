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
