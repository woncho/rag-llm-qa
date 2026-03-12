package llama;

/**

- Key-Value cache for autoregressive inference.
- 
- For each layer and each token position, we cache the projected key and value
- vectors so that we never recompute them for previously seen tokens.
- 
- Layout: cache[layer][pos * kvDim … (pos+1)*kvDim - 1]
  */
  public class KVCache {
  
  private final float[][] keyCache;   // [nLayers][seqLen * kvDim]
  private final float[][] valCache;   // [nLayers][seqLen * kvDim]
  private final int kvDim;
  private final int seqLen;
  
  public KVCache(ModelConfig cfg) {
  this.kvDim  = cfg.kvDim;
  this.seqLen = cfg.seqLen;
  keyCache = new float[cfg.nLayers][cfg.seqLen * cfg.kvDim];
  valCache = new float[cfg.nLayers][cfg.seqLen * cfg.kvDim];
  }
  
  /** Store key vector at (layer, pos). */
  public void storeKey(int layer, int pos, float[] key) {
  System.arraycopy(key, 0, keyCache[layer], pos * kvDim, kvDim);
  }
  
  /** Store value vector at (layer, pos). */
  public void storeVal(int layer, int pos, float[] val) {
  System.arraycopy(val, 0, valCache[layer], pos * kvDim, kvDim);
  }
  
  /** Get raw key cache row for layer (used during attention). */
  public float[] keyCacheRow(int layer) {
  return keyCache[layer];
  }
  
  /** Get raw value cache row for layer (used during attention). */
  public float[] valCacheRow(int layer) {
  return valCache[layer];
  }
  
  /** Zero out cache (for new generation runs). */
  public void clear() {
  for (int l = 0; l < keyCache.length; l++) {
  java.util.Arrays.fill(keyCache[l], 0f);
  java.util.Arrays.fill(valCache[l], 0f);
  }
  }
  }
