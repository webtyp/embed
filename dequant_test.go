package embed

import (
	"math"
	"testing"

	"webtyp.com/weights"
)

func TestDequantRow_RoundTrip(t *testing.T) {
	// Raw int8 row: [-128, -64, 0, 64, 127]
	// Scale: 0.01
	// Expected float32: [-1.28, -0.64, 0.0, 0.64, 1.27]
	raw := []byte{0x80, 0xC0, 0, 64, 127}
	scales := []float32{0.01}

	tensor := weights.Tensor{
		Data:   raw,
		Shape:  []int{1, 5},
		Scales: scales,
		DType:  weights.Int8,
	}

	dst := make([]float32, 5)
	dequantRow(tensor, 0, dst)

	expected := []float32{-1.28, -0.64, 0.0, 0.64, 1.27}
	for i, v := range dst {
		if math.Abs(float64(v-expected[i])) > 1e-5 {
			t.Errorf("at index %d: got %f, want %f", i, v, expected[i])
		}
	}
}

func TestDequantFull(t *testing.T) {
	// Float32 tensor bypass
	fData := []float32{1.5, 2.5, 3.5}
	byteData := make([]byte, len(fData)*4)
	for i, f := range fData {
		u := math.Float32bits(f)
		byteData[i*4] = byte(u)
		byteData[i*4+1] = byte(u >> 8)
		byteData[i*4+2] = byte(u >> 16)
		byteData[i*4+3] = byte(u >> 24)
	}

	tensorF32 := weights.Tensor{
		Data:  byteData,
		Shape: []int{1, 3},
		DType: weights.Float32,
	}

	outF32, err := dequantFull(tensorF32)
	if err != nil {
		t.Fatalf("dequantFull failed: %v", err)
	}
	for i, v := range outF32 {
		if v != fData[i] {
			t.Errorf("Float32 mismatch at %d: got %f, want %f", i, v, fData[i])
		}
	}

	// Int8 tensor dequantization: [10, -20, 30, -40]
	// -20 = 0xEC (236), -40 = 0xD8 (216)
	rawInt8 := []byte{10, 0xEC, 30, 0xD8}
	tensorInt8 := weights.Tensor{
		Data:   rawInt8,
		Shape:  []int{2, 2},
		Scales: []float32{0.1, 0.5},
		DType:  weights.Int8,
	}

	outInt8, err := dequantFull(tensorInt8)
	if err != nil {
		t.Fatalf("dequantFull failed for int8: %v", err)
	}
	expectedInt8 := []float32{1.0, -2.0, 15.0, -20.0}
	for i, v := range outInt8 {
		if math.Abs(float64(v-expectedInt8[i])) > 1e-5 {
			t.Errorf("Int8 mismatch at %d: got %f, want %f", i, v, expectedInt8[i])
		}
	}
}
