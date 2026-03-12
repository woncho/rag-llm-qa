package llama;

import java.util.function.Consumer;

/**

- High-level text generation runner.
- 
- Wraps the Transformer + Tokenizer + Sampler into a clean generation API
- supporting both streaming (token-by-token callback) and batch generation.
  */
  public class LlamaRunner {
  
  private final Transformer transformer;
  private final Tokenizer   tokenizer;
  private final Sampler     sampler;
  
  public LlamaRunner(Transformer transformer, Tokenizer tokenizer, Sampler sampler) {
  this.transformer = transformer;
  this.tokenizer   = tokenizer;
  this.sampler     = sampler;
  }
  
  // –– Generation ––
  
  /**
  - Generate text from a prompt.
  - 
  - @param prompt     input text
  - @param maxNewTokens max tokens to generate
  - @param temperature  sampling temperature (0 → greedy)
  - @param topP         nucleus sampling threshold
  - @param onToken      callback called for each generated token string (streaming)
  - @return full generated string (excluding prompt)
    */
    public String generate(String prompt,
    int maxNewTokens,
    float temperature,
    float topP,
    Consumer<String> onToken) {
    
    transformer.reset();
    int[] promptTokens = tokenizer.encode(prompt, true, false);
    int numPromptTokens = promptTokens.length;
    
    StringBuilder output = new StringBuilder();
    int token   = Tokenizer.BOS;
    int pos     = 0;
    int prevToken = Tokenizer.BOS;
    
    long startMs = System.currentTimeMillis();
    int generated = 0;
    
    while (pos < numPromptTokens + maxNewTokens - 1) {
    // During prompt processing use the prompt tokens;
    // after the prompt, use the last sampled token.
    int inputToken = (pos < numPromptTokens) ? promptTokens[pos] : token;
    
    ```
     float[] logits = transformer.forward(inputToken, pos);
    
     // We only sample after consuming all prompt tokens
     if (pos < numPromptTokens - 1) {
         // Teacher-forcing the prompt; next token is the next prompt token
         token = promptTokens[pos + 1];
     } else {
         // Sample from the model
         token = sampler.sample(logits.clone(), temperature, topP);
         generated++;
    
         if (token == Tokenizer.EOS) break;
    
         String piece = tokenizer.decode(prevToken, token);
         output.append(piece);
         if (onToken != null) onToken.accept(piece);
     }
    
     prevToken = inputToken;
     pos++;
    ```
    
    }
    
    long elapsed = System.currentTimeMillis() - startMs;
    if (generated > 0) {
    double tps = generated / (elapsed / 1000.0);
    System.err.printf(”%n[LlamaRunner] %d tokens in %.2fs → %.1f tok/s%n”,
    generated, elapsed / 1000.0, tps);
    }
    
    return output.toString();
    }
  
  /** Convenience overload: greedy decoding, no streaming. */
  public String generate(String prompt, int maxNewTokens) {
  return generate(prompt, maxNewTokens, 0f, 1.0f, null);
  }
  
  /** Streaming to stdout. */
  public String generateStreaming(String prompt, int maxNewTokens,
  float temperature, float topP) {
  System.out.print(prompt);
  System.out.flush();
  return generate(prompt, maxNewTokens, temperature, topP, piece -> {
  System.out.print(piece);
  System.out.flush();
  });
  }
  }
