package llama;

/**

- LLaMA model hyperparameters.
- Matches the config stored in GGUF / llama2.c format.
  */
  public class ModelConfig {
  public final int dim;           // transformer input/output dimension
  public final int hiddenDim;     // feedforward hidden dimension
  public final int nLayers;       // number of transformer layers
  public final int nHeads;        // number of query heads
  public final int nKvHeads;      // number of key/value heads (GQA)
  public final int vocabSize;     // vocabulary size
  public final int seqLen;        // max sequence length (context window)
  public final float normEps;     // RMSNorm epsilon
  
  // Derived
  public final int headDim;       // dim / nHeads
  public final int kvDim;         // dim * nKvHeads / nHeads  (GQA key/value dim)
  public final int kvMul;         // nHeads / nKvHeads (GQA multiplier)
  
  public ModelConfig(int dim, int hiddenDim, int nLayers, int nHeads,
  int nKvHeads, int vocabSize, int seqLen, float normEps) {
  this.dim = dim;
  this.hiddenDim = hiddenDim;
  this.nLayers = nLayers;
  this.nHeads = nHeads;
  this.nKvHeads = nKvHeads == 0 ? nHeads : nKvHeads;
  this.vocabSize = Math.abs(vocabSize);
  this.seqLen = seqLen;
  this.normEps = normEps;
  this.headDim = dim / nHeads;
  this.kvDim = (dim * this.nKvHeads) / nHeads;
  this.kvMul = nHeads / this.nKvHeads;
  }
  
  // –– Preset factory methods ––
  
  /** LLaMA-2 7B */
  public static ModelConfig llama2_7B() {
  return new ModelConfig(4096, 11008, 32, 32, 32, 32000, 4096, 1e-5f);
  }
  
  /** LLaMA-2 13B */
  public static ModelConfig llama2_13B() {
  return new ModelConfig(5120, 13824, 40, 40, 40, 32000, 4096, 1e-5f);
  }
  
  /** Tiny LLaMA (stories15M – used for testing without a real checkpoint) */
  public static ModelConfig tinyLlama() {
  return new ModelConfig(288, 768, 6, 6, 6, 32000, 256, 1e-5f);
  }
  
  /** LLaMA-3 8B (GQA: 32 Q heads, 8 KV heads) */
  public static ModelConfig llama3_8B() {
  return new ModelConfig(4096, 14336, 32, 32, 8, 128256, 8192, 1e-5f);
  }
  
  @Override
  public String toString() {
  return String.format(
  “ModelConfig{dim=%d, hidden=%d, layers=%d, heads=%d, kvHeads=%d, vocab=%d, seq=%d}”,
  dim, hiddenDim, nLayers, nHeads, nKvHeads, vocabSize, seqLen);
  }
  }
