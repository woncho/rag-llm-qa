package llama;

import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

/**

- Unit tests for core LLaMA Java components.
  */
  public class LlamaTest {
  
  // –– Tensor / math tests ––
  
  @Test
  void testRmsnorm() {
  float[] x      = {1f, 2f, 3f, 4f};
  float[] weight = {1f, 1f, 1f, 1f};
  float[] out    = new float[4];
  Tensor.rmsnorm(out, x, weight, 4, 1e-5f);
  
  ```
   // rms = sqrt((1+4+9+16)/4) = sqrt(7.5)
   double rms   = Math.sqrt(7.5);
   double scale = 1.0 / rms;
   assertEquals(1f * scale, out[0], 1e-4);
   assertEquals(2f * scale, out[1], 1e-4);
   assertEquals(3f * scale, out[2], 1e-4);
   assertEquals(4f * scale, out[3], 1e-4);
  ```
  
  }
  
  @Test
  void testSoftmax() {
  float[] x = {1f, 2f, 3f};
  Tensor.softmax(x, 0, 3);
  float sum = 0f;
  for (float v : x) sum += v;
  assertEquals(1.0f, sum, 1e-5f);
  // Verify ordering preserved
  assertTrue(x[2] > x[1]);
  assertTrue(x[1] > x[0]);
  }
  
  @Test
  void testSilu() {
  // SiLU(0) = 0
  assertEquals(0f, Tensor.silu(0f), 1e-6f);
  // SiLU(x) > 0 for x > 0
  assertTrue(Tensor.silu(1f) > 0);
  // SiLU is approximately linear for large x
  float large = Tensor.silu(10f);
  assertTrue(large > 9f);
  }
  
  @Test
  void testVecMatMul() {
  float[] vec = {1f, 2f};
  float[] mat = {1f, 0f, 0f, 1f}; // identity 2x2
  float[] out = new float[2];
  Tensor.vecMatMul(vec, mat, out, 2, 2);
  assertEquals(1f, out[0], 1e-6f);
  assertEquals(2f, out[1], 1e-6f);
  }
  
  // –– RoPE tests ––
  
  @Test
  void testRopePreservesNorm() {
  RoPE rope = new RoPE(128, 64);
  float[] vec = new float[64];
  java.util.Random r = new java.util.Random(42);
  for (int i = 0; i < vec.length; i++) vec[i] = r.nextFloat();
  float normBefore = norm(vec);
  rope.apply(vec, 5, 1);
  float normAfter = norm(vec);
  // RoPE is a rotation → norm preserved
  assertEquals(normBefore, normAfter, 1e-4f);
  }
  
  // –– Sampler tests ––
  
  @Test
  void testGreedySampler() {
  Sampler s = new Sampler(0);
  float[] logits = {0.1f, 5.0f, 0.2f, 0.3f};
  assertEquals(1, s.greedy(logits));
  }
  
  @Test
  void testTemperatureSampling() {
  Sampler s = new Sampler(42);
  float[] logits = {1f, 1f, 1f, 100f};
  // With low temperature the high-logit token dominates
  int count = 0;
  for (int i = 0; i < 100; i++) {
  int tok = s.sample(logits.clone(), 0.1f, 1.0f);
  if (tok == 3) count++;
  }
  assertTrue(count > 90, “Expected high-logit token to dominate: “ + count);
  }
  
  @Test
  void testSamplerDistribution() {
  Sampler s = new Sampler(99);
  // Uniform logits → each token should get ~25% with temp=1
  int[] counts = new int[4];
  for (int i = 0; i < 4000; i++) {
  float[] logits = {1f, 1f, 1f, 1f};
  counts[s.sample(logits, 1.0f, 1.0f)]++;
  }
  for (int c : counts) {
  assertTrue(c > 800 && c < 1200,
  “Expected ~1000 ± 200 per token, got “ + c);
  }
  }
  
  // –– Tokenizer tests ––
  
  @Test
  void testTokenizerEncodeDecode() {
  // Build a tiny vocab
  String[] vocab  = {”<unk>”, “<s>”, “</s>”, “\u2581hello”, “\u2581world”};
  float[]  scores = {0f, 0f, 0f, 1f, 2f};
  Tokenizer tok = new Tokenizer(vocab, scores);
  
  ```
   // "hello world" should encode to BOS + hello + world (roughly)
   int[] ids = tok.encode("\u2581hello\u2581world", false, false);
   assertTrue(ids.length > 0);
  ```
  
  }
  
  // –– ModelConfig tests ––
  
  @Test
  void testModelConfigDerived() {
  ModelConfig cfg = ModelConfig.tinyLlama();
  assertEquals(cfg.dim / cfg.nHeads, cfg.headDim);
  assertEquals(cfg.dim * cfg.nKvHeads / cfg.nHeads, cfg.kvDim);
  }
  
  @Test
  void testGQAConfig() {
  // LLaMA-3 8B: 32 Q heads, 8 KV heads
  ModelConfig cfg = ModelConfig.llama3_8B();
  assertEquals(32 / 8, cfg.kvMul);
  assertEquals(cfg.dim * 8 / 32, cfg.kvDim);
  }
  
  // –– Forward pass smoke test ––
  
  @Test
  void testForwardPassShape() {
  ModelConfig cfg = ModelConfig.tinyLlama();
  ModelWeights w  = new ModelWeights(cfg);
  // All-ones weights (extreme but runs without NaN)
  java.util.Arrays.fill(w.rmsFinalWeight, 1f);
  for (int l = 0; l < cfg.nLayers; l++) {
  java.util.Arrays.fill(w.rmsAttWeight[l], 1f);
  java.util.Arrays.fill(w.rmsFfnWeight[l], 1f);
  }
  
  ```
   Transformer t = new Transformer(w);
   float[] logits = t.forward(1, 0);
   assertEquals(cfg.vocabSize, logits.length);
   // All-zero weights → logits should be finite
   for (float v : logits) assertFalse(Float.isNaN(v), "NaN in logits");
  ```
  
  }
  
  // –– Helpers ––
  
  private float norm(float[] v) {
  float s = 0f;
  for (float x : v) s += x * x;
  return (float) Math.sqrt(s);
  }
  }
