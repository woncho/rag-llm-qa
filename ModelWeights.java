package llama;

/**

- All learned weights for a LLaMA model.
- 
- Naming convention follows the original llama2.c / Meta naming:
- token_embedding_table  – [vocab, dim]
- rms_att_weight[l]      – [dim]          (pre-attention RMSNorm)
- wq[l]                  – [dim, dim]     (query projection)
- wk[l]                  – [dim, kvDim]   (key projection)
- wv[l]                  – [dim, kvDim]   (value projection)
- wo[l]                  – [dim, dim]     (output projection)
- rms_ffn_weight[l]      – [dim]          (pre-FFN RMSNorm)
- w1[l]                  – [dim, hiddenDim]  (FFN gate)
- w2[l]                  – [hiddenDim, dim]  (FFN down)
- w3[l]                  – [dim, hiddenDim]  (FFN up / SwiGLU)
- rms_final_weight       – [dim]          (final RMSNorm)
- wcls                   – [dim, vocab]   (LM head; may be shared with embedding)
  */
  public class ModelWeights {
  public final ModelConfig cfg;

```
// Embedding
public final float[] tokenEmbeddingTable; // [vocab * dim]

// Per-layer weights (indexed by layer)
public final float[][] rmsAttWeight;   // [nLayers][dim]
public final float[][] wq;             // [nLayers][dim * dim]
public final float[][] wk;             // [nLayers][dim * kvDim]
public final float[][] wv;             // [nLayers][dim * kvDim]
public final float[][] wo;             // [nLayers][dim * dim]
public final float[][] rmsFfnWeight;   // [nLayers][dim]
public final float[][] w1;             // [nLayers][dim * hiddenDim]
public final float[][] w2;             // [nLayers][hiddenDim * dim]
public final float[][] w3;             // [nLayers][dim * hiddenDim]

// Final norm + LM head
public final float[] rmsFinalWeight;   // [dim]
public final float[] wcls;             // [vocab * dim]  (may alias tokenEmbeddingTable)

public ModelWeights(ModelConfig cfg) {
    this.cfg = cfg;
    int d = cfg.dim, h = cfg.hiddenDim, v = cfg.vocabSize;
    int kv = cfg.kvDim, L = cfg.nLayers;

    tokenEmbeddingTable = new float[v * d];

    rmsAttWeight = new float[L][d];
    wq           = new float[L][d * d];
    wk           = new float[L][d * kv];
    wv           = new float[L][d * kv];
    wo           = new float[L][d * d];
    rmsFfnWeight = new float[L][d];
    w1           = new float[L][d * h];
    w2           = new float[L][h * d];
    w3           = new float[L][d * h];

    rmsFinalWeight = new float[d];
    wcls           = tokenEmbeddingTable; // tied weights (LLaMA-2 default)
}

/**
 * Construct weights with a separate (un-tied) LM head.
 */
public ModelWeights(ModelConfig cfg, boolean tiedWeights) {
    this(cfg);
    // If not tied, wcls field can't be re-assigned (final), so subclass or use builder.
    // This constructor is a placeholder for the tied=true case.
    if (!tiedWeights) {
        throw new UnsupportedOperationException(
            "Use ModelWeightsBuilder for untied LM head weights.");
    }
}

/** Lookup token embedding row: returns view slice [dim] */
public float[] getEmbedding(int tokenId) {
    float[] row = new float[cfg.dim];
    System.arraycopy(tokenEmbeddingTable, tokenId * cfg.dim, row, 0, cfg.dim);
    return row;
}
```

}
