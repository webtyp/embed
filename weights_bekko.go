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

func loadLayer(get func(string) (weights.Tensor, error), i int) (transformer.LayerWeights, error) {
	prefix := fmt.Sprintf("layers.%d.", i)

	wqkv, err := get(fmt.Sprintf("%sattn.Wqkv.weight", prefix))
	if err != nil {
		return transformer.LayerWeights{}, err
	}
	wqkvF, err := dequantFull(wqkv)
	if err != nil {
		return transformer.LayerWeights{}, err
	}

	wo, err := get(fmt.Sprintf("%sattn.Wo.weight", prefix))
	if err != nil {
		return transformer.LayerWeights{}, err
	}
	woF, err := dequantFull(wo)
	if err != nil {
		return transformer.LayerWeights{}, err
	}

	wi, err := get(fmt.Sprintf("%smlp.Wi.weight", prefix))
	if err != nil {
		return transformer.LayerWeights{}, err
	}
	wiF, err := dequantFull(wi)
	if err != nil {
		return transformer.LayerWeights{}, err
	}

	mlpWo, err := get(fmt.Sprintf("%smlp.Wo.weight", prefix))
	if err != nil {
		return transformer.LayerWeights{}, err
	}
	mlpWoF, err := dequantFull(mlpWo)
	if err != nil {
		return transformer.LayerWeights{}, err
	}

	mlpNorm, err := get(fmt.Sprintf("%smlp_norm.weight", prefix))
	if err != nil {
		return transformer.LayerWeights{}, err
	}
	mlpNormF, err := dequantFull(mlpNorm)
	if err != nil {
		return transformer.LayerWeights{}, err
	}

	var attnNormF []float32
	attnNorm, err := get(fmt.Sprintf("%sattn_norm.weight", prefix))
	if err == nil {
		attnNormF, err = dequantFull(attnNorm)
		if err != nil {
			return transformer.LayerWeights{}, err
		}
	}

	return transformer.LayerWeights{
		WqkvT:         wqkvF,
		WoT:           woF,
		WiT:           wiF,
		MlpWoT:        mlpWoF,
		AttnNormGamma: attnNormF,
		MlpNormGamma:  mlpNormF,
	}, nil
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
