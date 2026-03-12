package llama;

import java.util.Arrays;

/**

- LLaMA Transformer — forward pass implementation.
- 
- Architecture:
- Embed → N × (RMSNorm → GQA Self-Attention w/ RoPE → residual
- ```
             → RMSNorm → SwiGLU FFN → residual)
  ```
- → RMSNorm → LM head
- 
- All operations are pure Java float arithmetic; no external libraries required.
  */
  public class Transformer {
  
  private final ModelConfig cfg;
  private final ModelWeights w;
  private final KVCache kvCache;
  private final RoPE rope;
  
  // Scratch buffers (reused each forward call to avoid GC pressure)
  private final float[] x;       // [dim]  current hidden state
  private final float[] xb;      // [dim]  scratch
  private final float[] xb2;     // [dim]  scratch
  private final float[] q;       // [nHeads * headDim]
  private final float[] k;       // [nKvHeads * headDim]
  private final float[] v;       // [nKvHeads * headDim]
  private final float[] att;     // [nHeads * seqLen]
  private final float[] hb;      // [hiddenDim]
  private final float[] hb2;     // [hiddenDim]
  private final float[] logits;  // [vocabSize]
  
  public Transformer(ModelWeights weights) {
  this.cfg     = weights.cfg;
  this.w       = weights;
  this.kvCache = new KVCache(cfg);
  this.rope    = new RoPE(cfg.seqLen, cfg.headDim);
  
  ```
   x      = new float[cfg.dim];
   xb     = new float[cfg.dim];
   xb2    = new float[cfg.dim];
   q      = new float[cfg.nHeads   * cfg.headDim];
   k      = new float[cfg.nKvHeads * cfg.headDim];
   v      = new float[cfg.nKvHeads * cfg.headDim];
   att    = new float[cfg.nHeads   * cfg.seqLen];
   hb     = new float[cfg.hiddenDim];
   hb2    = new float[cfg.hiddenDim];
   logits = new float[cfg.vocabSize];
  ```
  
  }
  
  /**
  - Run one forward step.
  - 
  - @param token current token id
  - @param pos   current position in sequence (0-indexed)
  - @return      raw logits over vocabulary
    */
    public float[] forward(int token, int pos) {
    final int dim     = cfg.dim;
    final int kvDim   = cfg.kvDim;
    final int headDim = cfg.headDim;
    final int kvMul   = cfg.kvMul;
    final float scale = (float) (1.0 / Math.sqrt(headDim));
    
    // 1. Token embedding
    System.arraycopy(w.tokenEmbeddingTable, token * dim, x, 0, dim);
    
    // 2. Transformer layers
    for (int l = 0; l < cfg.nLayers; l++) {
    
    ```
     // --- 2a. Attention block ---
     // Pre-attention RMSNorm
     Tensor.rmsnorm(xb, x, w.rmsAttWeight[l], dim, cfg.normEps);
    
     // Project Q, K, V
     Tensor.vecMatMul(xb, w.wq[l], q, dim, dim);
     Tensor.vecMatMul(xb, w.wk[l], k, dim, kvDim);
     Tensor.vecMatMul(xb, w.wv[l], v, dim, kvDim);
    
     // Apply RoPE
     rope.apply(q, pos, cfg.nHeads);
     rope.apply(k, pos, cfg.nKvHeads);
    
     // Store K, V in cache
     kvCache.storeKey(l, pos, k);
     kvCache.storeVal(l, pos, v);
    
     // Multi-head (grouped query) attention
     float[] kCache = kvCache.keyCacheRow(l);
     float[] vCache = kvCache.valCacheRow(l);
    
     for (int h = 0; h < cfg.nHeads; h++) {
         int qOff    = h * headDim;
         int kvHead  = h / kvMul;         // GQA: which KV head this Q head uses
         int attOff  = h * cfg.seqLen;
    
         // Attention scores: q · k^T / sqrt(headDim)
         for (int t = 0; t <= pos; t++) {
             float score = 0f;
             int kOff = t * kvDim + kvHead * headDim;
             for (int i = 0; i < headDim; i++) {
                 score += q[qOff + i] * kCache[kOff + i];
             }
             att[attOff + t] = score * scale;
         }
    
         // Softmax over [0..pos]
         Tensor.softmax(att, attOff, pos + 1);
    
         // Weighted sum of values
         int xbOff = h * headDim;
         Arrays.fill(xb, xbOff, xbOff + headDim, 0f);
         for (int t = 0; t <= pos; t++) {
             float a   = att[attOff + t];
             int vOff  = t * kvDim + kvHead * headDim;
             for (int i = 0; i < headDim; i++) {
                 xb[xbOff + i] += a * vCache[vOff + i];
             }
         }
     }
    
     // Output projection
     Tensor.vecMatMul(xb, w.wo[l], xb2, dim, dim);
    
     // Residual connection
     Tensor.addInPlace(x, xb2, dim);
    
     // --- 2b. Feed-forward block ---
     // Pre-FFN RMSNorm
     Tensor.rmsnorm(xb, x, w.rmsFfnWeight[l], dim, cfg.normEps);
    
     // SwiGLU: out = (W1·x ⊙ SiLU(W3·x)) * W2
     Tensor.vecMatMul(xb, w.w1[l], hb,  dim, cfg.hiddenDim);
     Tensor.vecMatMul(xb, w.w3[l], hb2, dim, cfg.hiddenDim);
    
     for (int i = 0; i < cfg.hiddenDim; i++) {
         hb[i] = Tensor.silu(hb[i]) * hb2[i];
     }
    
     Tensor.vecMatMul(hb, w.w2[l], xb, cfg.hiddenDim, dim);
    
     // Residual connection
     Tensor.addInPlace(x, xb, dim);
    ```
    
    }
    
    // 3. Final RMSNorm
    Tensor.rmsnorm(x, x, w.rmsFinalWeight, dim, cfg.normEps);
    
    // 4. LM head (project to vocabulary)
    Tensor.vecMatMul(x, w.wcls, logits, dim, cfg.vocabSize);
    
    return logits;
    }
  
  /** Reset KV cache (start fresh generation). */
  public void reset() {
  kvCache.clear();
  }
  
  public ModelConfig getConfig() { return cfg; }
  public KVCache getKVCache()    { return kvCache; }
  }
