package llama;

import java.util.Arrays;
import java.util.Random;

/**

- Token sampling strategies for LLaMA inference.
- 
- - Greedy (argmax)
- - Temperature scaling
- - Top-K
- - Top-P (nucleus sampling)
- - Combined temperature + top-p (typical generation)
    */
    public class Sampler {
  
  private final Random rng;
  
  public Sampler(long seed) {
  this.rng = new Random(seed);
  }
  
  public Sampler() {
  this(System.currentTimeMillis());
  }
  
  // –– Public API ––
  
  /** Greedy decoding: argmax over logits. */
  public int greedy(float[] logits) {
  int best = 0;
  for (int i = 1; i < logits.length; i++) {
  if (logits[i] > logits[best]) best = i;
  }
  return best;
  }
  
  /**
  - Sample with temperature, optional top-p nucleus filtering.
  - 
  - @param logits raw logits (will be modified in-place)
  - @param temp   temperature (≤0 → greedy, 1.0 → no scaling)
  - @param topP   nucleus probability (0.9 is common; ≥1.0 disables)
    */
    public int sample(float[] logits, float temp, float topP) {
    if (temp <= 0f) return greedy(logits);
    
    // Apply temperature
    int n = logits.length;
    for (int i = 0; i < n; i++) logits[i] /= temp;
    
    // Softmax
    Tensor.softmax(logits, 0, n);
    
    // Top-p nucleus filtering
    if (topP < 1.0f) {
    return sampleTopP(logits, topP);
    }
    return sampleMultinomial(logits, n);
    }
  
  /**
  - Convenience: sample with temperature only (no top-p).
    */
    public int sample(float[] logits, float temp) {
    return sample(logits, temp, 1.0f);
    }
  
  // –– Private helpers ––
  
  private int sampleMultinomial(float[] probs, int n) {
  float r = rng.nextFloat();
  float cdf = 0f;
  for (int i = 0; i < n; i++) {
  cdf += probs[i];
  if (r < cdf) return i;
  }
  return n - 1;
  }
  
  private int sampleTopP(float[] probs, float topP) {
  int n = probs.length;
  
  ```
  // Build (prob, index) pairs and sort descending by prob
  int[] idx = new int[n];
  for (int i = 0; i < n; i++) idx[i] = i;
  // Partial sort: sort indices by probs descending
  // Use insertion sort on a cutoff to keep it O(n log n) worst-case
  Integer[] idxBoxed = new Integer[n];
  for (int i = 0; i < n; i++) idxBoxed[i] = i;
  Arrays.sort(idxBoxed, (a, b) -> Float.compare(probs[b], probs[a]));
  
  // Find nucleus cutoff
  float cumsum = 0f;
  int cutoff = 0;
  for (; cutoff < n; cutoff++) {
      cumsum += probs[idxBoxed[cutoff]];
      if (cumsum >= topP) { cutoff++; break; }
  }
  if (cutoff == 0) cutoff = 1;
  
  // Sample within nucleus
  float r = rng.nextFloat() * cumsum;
  float cdf = 0f;
  for (int i = 0; i < cutoff; i++) {
      cdf += probs[idxBoxed[i]];
      if (r < cdf) return idxBoxed[i];
  }
  return idxBoxed[cutoff - 1];
  ```
  
  }
  
  /**
  - Top-K sampling: keep only the K highest-probability tokens.
    */
    public int sampleTopK(float[] logits, float temp, int k) {
    if (temp <= 0f) return greedy(logits);
    int n = logits.length;
    for (int i = 0; i < n; i++) logits[i] /= temp;
    Tensor.softmax(logits, 0, n);
    
    // Zero out everything outside top-k
    Integer[] idx = new Integer[n];
    for (int i = 0; i < n; i++) idx[i] = i;
    Arrays.sort(idx, (a, b) -> Float.compare(logits[b], logits[a]));
    for (int i = k; i < n; i++) logits[idx[i]] = 0f;
    
    // Renormalize and sample
    float sum = 0f;
    for (int i = 0; i < k; i++) sum += logits[idx[i]];
    float r = rng.nextFloat() * sum;
    float cdf = 0f;
    for (int i = 0; i < k; i++) {
    cdf += logits[idx[i]];
    if (r < cdf) return idx[i];
    }
    return idx[k - 1];
    }
    }
