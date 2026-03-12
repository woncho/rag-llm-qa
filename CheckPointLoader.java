package llama;

import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.nio.file.*;

/**

- Loads model weights from a llama2.c-style binary checkpoint file.
- 
- File format (all little-endian):
- Header (7 × int32):
- ```
  dim, hiddenDim, nLayers, nHeads, nKvHeads, vocabSize, seqLen
  ```
- Body (float32 arrays in order):
- ```
  token_embedding_table  [vocabSize × dim]   (or [|vocabSize| × dim] if shared)
  ```
- ```
  rms_att_weight[0..L-1] [dim]
  ```
- ```
  wq[0..L-1]             [dim × dim]
  ```
- ```
  wk[0..L-1]             [dim × kvDim]
  ```
- ```
  wv[0..L-1]             [dim × kvDim]
  ```
- ```
  wo[0..L-1]             [dim × dim]
  ```
- ```
  rms_ffn_weight[0..L-1] [dim]
  ```
- ```
  w1[0..L-1]             [dim × hiddenDim]
  ```
- ```
  w2[0..L-1]             [hiddenDim × dim]
  ```
- ```
  w3[0..L-1]             [dim × hiddenDim]
  ```
- ```
  rms_final_weight       [dim]
  ```
- ```
  freq_cis_real          [seqLen × headDim/2]  (ignored – we compute RoPE)
  ```
- ```
  freq_cis_imag          [seqLen × headDim/2]  (ignored)
  ```
- ```
  wcls                   [vocabSize × dim]  (only if vocabSize < 0 → shared weights)
  ```

*/
public class CheckpointLoader {

```
/**
 * Load a llama2.c .bin checkpoint and return a populated ModelWeights.
 */
public static ModelWeights load(String checkpointPath) throws IOException {
    Path path = Paths.get(checkpointPath);
    ByteBuffer buf = ByteBuffer
        .wrap(Files.readAllBytes(path))
        .order(ByteOrder.LITTLE_ENDIAN);

    // --- Read header ---
    int dim        = buf.getInt();
    int hiddenDim  = buf.getInt();
    int nLayers    = buf.getInt();
    int nHeads     = buf.getInt();
    int nKvHeads   = buf.getInt();
    int vocabSizeRaw = buf.getInt(); // negative means shared weights
    int seqLen     = buf.getInt();

    boolean sharedWeights = vocabSizeRaw > 0;
    ModelConfig cfg = new ModelConfig(dim, hiddenDim, nLayers, nHeads,
                                      nKvHeads, vocabSizeRaw, seqLen, 1e-5f);

    System.out.println("[CheckpointLoader] " + cfg);

    ModelWeights w = new ModelWeights(cfg);

    // --- Read weights ---
    readFloats(buf, w.tokenEmbeddingTable);

    for (int l = 0; l < nLayers; l++) readFloats(buf, w.rmsAttWeight[l]);
    for (int l = 0; l < nLayers; l++) readFloats(buf, w.wq[l]);
    for (int l = 0; l < nLayers; l++) readFloats(buf, w.wk[l]);
    for (int l = 0; l < nLayers; l++) readFloats(buf, w.wv[l]);
    for (int l = 0; l < nLayers; l++) readFloats(buf, w.wo[l]);
    for (int l = 0; l < nLayers; l++) readFloats(buf, w.rmsFfnWeight[l]);
    for (int l = 0; l < nLayers; l++) readFloats(buf, w.w1[l]);
    for (int l = 0; l < nLayers; l++) readFloats(buf, w.w2[l]);
    for (int l = 0; l < nLayers; l++) readFloats(buf, w.w3[l]);

    readFloats(buf, w.rmsFinalWeight);

    // Skip precomputed RoPE tables (seqLen * headDim/2 × 2 floats)
    int ropeFloats = seqLen * (dim / nHeads / 2) * 2;
    buf.position(buf.position() + ropeFloats * Float.BYTES);

    // wcls is aliased to tokenEmbeddingTable when weights are shared
    // (nothing to read – already populated above)
    // If NOT shared, read separate wcls
    if (!sharedWeights && buf.remaining() >= cfg.vocabSize * dim * Float.BYTES) {
        // ModelWeights ties wcls to tokenEmbeddingTable by default;
        // for untied we'd need a separate field – left as an exercise.
        System.out.println("[CheckpointLoader] Warning: untied wcls not separately stored " +
                           "in current ModelWeights; using embedding table.");
    }

    return w;
}

private static void readFloats(ByteBuffer buf, float[] dst) {
    buf.asFloatBuffer().get(dst);
    buf.position(buf.position() + dst.length * Float.BYTES);
}
```

}
