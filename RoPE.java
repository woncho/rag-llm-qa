package llama;

/**

- Rotary Position Embedding (RoPE).
- 
- For each head, the first headDim floats of Q and K are rotated in 2D pairs:
- [x0, x1] -> [x0*cos - x1*sin,  x0*sin + x1*cos]
- 
- Frequencies: theta_i = 1 / (base ^ (2i / headDim))
- 
- LLaMA-2 uses base=10000; LLaMA-3 uses base=500000 with scaled frequencies.
  */
  public class RoPE {
  
  private final float[] cosCache; // [seqLen * headDim/2]
  private final float[] sinCache;
  private final int headDim;
  private final int halfHead;
  
  public RoPE(int seqLen, int headDim, float base) {
  this.headDim = headDim;
  this.halfHead = headDim / 2;
  cosCache = new float[seqLen * halfHead];
  sinCache = new float[seqLen * halfHead];
  
  ```
   for (int pos = 0; pos < seqLen; pos++) {
       for (int i = 0; i < halfHead; i++) {
           double theta = pos / Math.pow(base, (2.0 * i) / headDim);
           cosCache[pos * halfHead + i] = (float) Math.cos(theta);
           sinCache[pos * halfHead + i] = (float) Math.sin(theta);
       }
   }
  ```
  
  }
  
  /** Default LLaMA-2 base */
  public RoPE(int seqLen, int headDim) {
  this(seqLen, headDim, 10000f);
  }
  
  /**
  - Apply RoPE in-place to a query or key vector for ALL heads.
  - 
  - @param vec    flat buffer of shape [nHeads * headDim] (or [nKvHeads * headDim])
  - @param pos    current token position
  - @param nHeads number of heads stored in vec
    */
    public void apply(float[] vec, int pos, int nHeads) {
    for (int h = 0; h < nHeads; h++) {
    int base = h * headDim;
    for (int i = 0; i < halfHead; i++) {
    float x0 = vec[base + i];
    float x1 = vec[base + i + halfHead];
    float c  = cosCache[pos * halfHead + i];
    float s  = sinCache[pos * halfHead + i];
    vec[base + i]           = x0 * c - x1 * s;
    vec[base + i + halfHead] = x0 * s + x1 * c;
    }
    }
    }
  
  /**
  - Interleaved variant (used by some HuggingFace RoPE implementations):
  - pairs are (vec[2i], vec[2i+1]) rather than (vec[i], vec[i + halfHead]).
    */
    public void applyInterleaved(float[] vec, int pos, int nHeads) {
    for (int h = 0; h < nHeads; h++) {
    int base = h * headDim;
    for (int i = 0; i < halfHead; i++) {
    float x0 = vec[base + 2 * i];
    float x1 = vec[base + 2 * i + 1];
    float c  = cosCache[pos * halfHead + i];
    float s  = sinCache[pos * halfHead + i];
    vec[base + 2 * i]     = x0 * c - x1 * s;
    vec[base + 2 * i + 1] = x0 * s + x1 * c;
    }
    }
    }
    
