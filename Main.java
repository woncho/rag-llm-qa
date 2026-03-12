package llama;

/**

- LLaMA in Pure Java — command-line entry point.
- 
- Usage:
- java -cp . llama.Main <checkpoint.bin> <tokenizer.bin> [options]
- 
- Options:
- –prompt   “text”    Input prompt (default: “Once upon a time”)
- –steps    N         Max new tokens to generate (default: 256)
- –temp     F         Temperature 0.0-2.0 (default: 0.8; 0 = greedy)
- –topp     F         Nucleus sampling threshold (default: 0.9)
- –seed     N         Random seed (default: time-based)
- –demo               Run built-in demo with random weights (no files needed)
  */
  public class Main {

```
public static void main(String[] args) throws Exception {
    // Default values
    String checkpointPath = null;
    String tokenizerPath  = null;
    String prompt         = "Once upon a time";
    int    steps          = 256;
    float  temperature    = 0.8f;
    float  topP           = 0.9f;
    long   seed           = System.currentTimeMillis();
    boolean demo          = false;

    // Parse args
    for (int i = 0; i < args.length; i++) {
        switch (args[i]) {
            case "--prompt": prompt         = args[++i]; break;
            case "--steps":  steps          = Integer.parseInt(args[++i]); break;
            case "--temp":   temperature    = Float.parseFloat(args[++i]); break;
            case "--topp":   topP           = Float.parseFloat(args[++i]); break;
            case "--seed":   seed           = Long.parseLong(args[++i]); break;
            case "--demo":   demo           = true; break;
            default:
                if (checkpointPath == null) checkpointPath = args[i];
                else if (tokenizerPath == null) tokenizerPath = args[i];
        }
    }

    if (demo || checkpointPath == null) {
        runDemo(prompt, steps, temperature, topP, seed);
    } else {
        runFromFiles(checkpointPath, tokenizerPath, prompt, steps,
                     temperature, topP, seed);
    }
}

// ---- Demo mode: random weights, built-in tiny config ----
private static void runDemo(String prompt, int steps, float temp, float topP, long seed)
        throws Exception {
    System.out.println("=== LLaMA Pure Java — Demo Mode (random weights) ===");
    System.out.println("Config: TinyLLaMA (288-dim, 6 layers, 6 heads)");
    System.out.println("NOTE: Output is gibberish — weights are random.");
    System.out.println();

    ModelConfig cfg = ModelConfig.tinyLlama();
    ModelWeights w  = new ModelWeights(cfg);
    initRandomWeights(w, seed);

    // Minimal vocab for demo
    String[] vocab  = buildDemoVocab(cfg.vocabSize);
    float[]  scores = new float[cfg.vocabSize];
    java.util.Random r = new java.util.Random(seed);
    for (int i = 0; i < scores.length; i++) scores[i] = r.nextFloat();

    Tokenizer   tok     = new Tokenizer(vocab, scores);
    Transformer trans   = new Transformer(w);
    Sampler     sampler = new Sampler(seed);
    LlamaRunner runner  = new LlamaRunner(trans, tok, sampler);

    System.out.println("Prompt: " + prompt);
    System.out.println("--- Generated output ---");
    runner.generateStreaming(prompt, steps, temp, topP);
    System.out.println("\n--- End ---");
}

private static void initRandomWeights(ModelWeights w, long seed) {
    java.util.Random r = new java.util.Random(seed);
    float scale = 0.02f;
    randomFill(w.tokenEmbeddingTable, r, scale);
    for (int l = 0; l < w.cfg.nLayers; l++) {
        fillOnes(w.rmsAttWeight[l]);
        randomFill(w.wq[l], r, scale);
        randomFill(w.wk[l], r, scale);
        randomFill(w.wv[l], r, scale);
        randomFill(w.wo[l], r, scale);
        fillOnes(w.rmsFfnWeight[l]);
        randomFill(w.w1[l], r, scale);
        randomFill(w.w2[l], r, scale);
        randomFill(w.w3[l], r, scale);
    }
    fillOnes(w.rmsFinalWeight);
}

private static void randomFill(float[] a, java.util.Random r, float scale) {
    for (int i = 0; i < a.length; i++) a[i] = (r.nextFloat() - 0.5f) * 2f * scale;
}

private static void fillOnes(float[] a) { java.util.Arrays.fill(a, 1f); }

private static String[] buildDemoVocab(int size) {
    String[] v = new String[size];
    // First few special tokens
    v[0] = "<unk>"; v[1] = "<s>"; v[2] = "</s>";
    // Fill rest with printable tokens
    String letters = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?";
    String[] words  = {"the","a","is","was","in","on","of","to","and","that",
                       "it","with","for","as","at","be","by","from","or","an",
                       "time","once","upon","long","ago","there","lived","great",
                       "king","queen","forest","river","mountain","star","light"};
    int wi = 0;
    for (int i = 3; i < size; i++) {
        if (wi < words.length) { v[i] = "\u2581" + words[wi++]; }
        else                   { v[i] = String.valueOf(letters.charAt(i % letters.length())); }
    }
    return v;
}

// ---- File mode ----
private static void runFromFiles(String ckpt, String tokPath,
                                 String prompt, int steps,
                                 float temp, float topP, long seed) throws Exception {
    System.out.println("Loading checkpoint: " + ckpt);
    ModelWeights w   = CheckpointLoader.load(ckpt);
    System.out.println("Loading tokenizer:  " + tokPath);
    Tokenizer    tok = Tokenizer.fromFile(tokPath, w.cfg.vocabSize);

    Transformer  trans  = new Transformer(w);
    Sampler      samp   = new Sampler(seed);
    LlamaRunner  runner = new LlamaRunner(trans, tok, samp);

    System.out.println("Generating...\n");
    runner.generateStreaming(prompt, steps, temp, topP);
    System.out.println();
}
```

}
