package llama;

import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.util.*;

/**

- Byte-Pair Encoding tokenizer compatible with LLaMA-2 / llama2.c tokenizer.bin format.
- 
- The binary file format (tokenizer.bin from llama2.c):
- [4 bytes: max_token_length int32]
- For each of vocabSize tokens:
- ```
  [4 bytes: score float32]
  ```
- ```
  [4 bytes: length int32]
  ```
- ```
  [length bytes: UTF-8 token string]
  ```
- 
- Encoding uses greedy BPE merge: find highest-scoring merge, apply, repeat.
- Decoding maps token ids → strings with ▁ (U+2581) → space substitution.
  */
  public class Tokenizer {
  
  private final String[] vocab;
  private final float[]  vocabScores;
  private final Map<String, Integer> vocabIndex;
  private final int vocabSize;
  
  // Special tokens
  public static final int BOS = 1; // <s>
  public static final int EOS = 2; // </s>
  
  // –– Construction ––
  
  public Tokenizer(String[] vocab, float[] vocabScores) {
  this.vocab       = vocab;
  this.vocabScores = vocabScores;
  this.vocabSize   = vocab.length;
  this.vocabIndex  = new HashMap<>(vocabSize * 2);
  for (int i = 0; i < vocabSize; i++) {
  vocabIndex.put(vocab[i], i);
  }
  }
  
  /**
  - Load tokenizer from a llama2.c-style tokenizer.bin file.
    */
    public static Tokenizer fromFile(String path, int vocabSize) throws IOException {
    ByteBuffer buf = ByteBuffer
    .wrap(Files.readAllBytes(Paths.get(path)))
    .order(ByteOrder.LITTLE_ENDIAN);
    
    buf.getInt(); // max_token_length (unused here)
    
    String[] vocab  = new String[vocabSize];
    float[]  scores = new float[vocabSize];
    
    for (int i = 0; i < vocabSize; i++) {
    scores[i] = buf.getFloat();
    int len   = buf.getInt();
    byte[] bytes = new byte[len];
    buf.get(bytes);
    vocab[i] = new String(bytes, “UTF-8”);
    }
    return new Tokenizer(vocab, scores);
    }
  
  // –– Encoding ––
  
  /**
  - Encode text to token ids (BPE).
  - 
  - @param text     input string
  - @param bos      prepend BOS token
  - @param eos      append EOS token
    */
    public int[] encode(String text, boolean bos, boolean eos) {
    // Start: every character (or special byte) is its own token
    List<Integer> tokens = new ArrayList<>();
    
    if (bos) tokens.add(BOS);
    
    // Prefix space like SentencePiece
    if (!text.isEmpty()) {
    Integer dummy = vocabIndex.get(” “ + text.charAt(0));
    if (dummy != null) {
    // handled inside loop
    }
    }
    
    // Convert each character to its token id
    for (int i = 0; i < text.length(); ) {
    // Try to match longest prefix
    String ch = String.valueOf(text.charAt(i));
    // SentencePiece uses ▁ for space
    if (ch.equals(” “)) ch = “\u2581”;
    Integer id = vocabIndex.get(ch);
    if (id == null) {
    // Byte fallback: encode as <0xXX>
    byte[] bytes = ch.getBytes(java.nio.charset.StandardCharsets.UTF_8);
    for (byte b : bytes) {
    String byteStr = String.format(”<0x%02X>”, b & 0xFF);
    id = vocabIndex.getOrDefault(byteStr, 3); // unk
    tokens.add(id);
    }
    } else {
    tokens.add(id);
    }
    i++;
    }
    
    // BPE merges: repeatedly find and merge the highest-scoring adjacent pair
    while (true) {
    float bestScore = Float.NEGATIVE_INFINITY;
    int   bestId    = -1;
    int   bestPos   = -1;
    
    ```
     for (int i = 0; i + 1 < tokens.size(); i++) {
         String merged = vocab[tokens.get(i)] + vocab[tokens.get(i + 1)];
         Integer id = vocabIndex.get(merged);
         if (id != null && vocabScores[id] > bestScore) {
             bestScore = vocabScores[id];
             bestId    = id;
             bestPos   = i;
         }
     }
    
     if (bestPos == -1) break; // no more merges possible
    
     tokens.set(bestPos, bestId);
     tokens.remove(bestPos + 1);
    ```
    
    }
    
    if (eos) tokens.add(EOS);
    
    return tokens.stream().mapToInt(Integer::intValue).toArray();
    }
  
  /** Encode with BOS prepended (common default). */
  public int[] encode(String text) {
  return encode(text, true, false);
  }
  
  // –– Decoding ––
  
  /**
  - Decode a single token id to its string representation.
  - Replaces ▁ with space and handles <0xXX> byte tokens.
    */
    public String decode(int prevToken, int token) {
    String piece = vocab[token];
    
    // After BOS, suppress leading space (SentencePiece convention)
    if (prevToken == BOS && piece.startsWith(”\u2581”)) {
    piece = piece.substring(1);
    }
    
    // ▁ → space
    piece = piece.replace(”\u2581”, “ “);
    
    // <0xXX> byte token → raw byte
    if (piece.startsWith(”<0x”) && piece.endsWith(”>”) && piece.length() == 6) {
    try {
    int byteVal = Integer.parseInt(piece.substring(3, 5), 16);
    return new String(new byte[]{(byte) byteVal},
    java.nio.charset.StandardCharsets.UTF_8);
    } catch (NumberFormatException ignored) {}
    }
    
    return piece;
    }
  
  /** Decode a full sequence of token ids to string. */
  public String decode(int[] tokens) {
  StringBuilder sb = new StringBuilder();
  for (int i = 0; i < tokens.length; i++) {
  sb.append(decode(i == 0 ? BOS : tokens[i - 1], tokens[i]));
  }
  return sb.toString();
  }
  
  public int vocabSize()            { return vocabSize; }
  public String getToken(int id)    { return vocab[id]; }
  public int getTokenId(String tok) { return vocabIndex.getOrDefault(tok, -1); }
  }
