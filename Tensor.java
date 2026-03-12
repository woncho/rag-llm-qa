package llama;

import java.util.Arrays;

/**

- A simple float tensor with shape support.
  */
  public class Tensor {
  public final float[] data;
  public final int[] shape;
  public final int size;
  
  public Tensor(int… shape) {
  this.shape = shape.clone();
  int total = 1;
  for (int d : shape) total *= d;
  this.size = total;
  this.data = new float[total];
  }
  
  public Tensor(float[] data, int… shape) {
  this.data = data;
  this.shape = shape.clone();
  int total = 1;
  for (int d : shape) total *= d;
  this.size = total;
  }
  
  public float get(int i) { return data[i]; }
  public void set(int i, float v) { data[i] = v; }
  
  /** Matrix multiply: [m, k] x [k, n] -> [m, n] */
  public static Tensor matmul(Tensor a, Tensor b) {
  int m = a.shape[0], k = a.shape[1], n = b.shape[1];
  Tensor out = new Tensor(m, n);
  for (int i = 0; i < m; i++) {
  for (int j = 0; j < n; j++) {
  float sum = 0f;
  for (int p = 0; p < k; p++) {
  sum += a.data[i * k + p] * b.data[p * n + j];
  }
  out.data[i * n + j] = sum;
  }
  }
  return out;
  }
  
  /** Batched matmul optimized for vector x matrix: [k] x [k, n] -> [n] */
  public static void vecMatMul(float[] vec, float[] mat, float[] out, int k, int n) {
  Arrays.fill(out, 0f);
  for (int i = 0; i < k; i++) {
  float vi = vec[i];
  if (vi == 0f) continue;
  int base = i * n;
  for (int j = 0; j < n; j++) {
  out[j] += vi * mat[base + j];
  }
  }
  }
  
  /** Element-wise add in place: a += b */
  public static void addInPlace(float[] a, float[] b, int len) {
  for (int i = 0; i < len; i++) a[i] += b[i];
  }
  
  /** Copy src into dst */
  public static void copy(float[] src, float[] dst, int len) {
  System.arraycopy(src, 0, dst, 0, len);
  }
  
  /** Softmax in place */
  public static void softmax(float[] x, int offset, int len) {
  float max = Float.NEGATIVE_INFINITY;
  for (int i = 0; i < len; i++) {
  if (x[offset + i] > max) max = x[offset + i];
  }
  float sum = 0f;
  for (int i = 0; i < len; i++) {
  x[offset + i] = (float) Math.exp(x[offset + i] - max);
  sum += x[offset + i];
  }
  for (int i = 0; i < len; i++) x[offset + i] /= sum;
  }
  
  /** RMS Norm: out[i] = x[i] / rms(x) * weight[i] */
  public static void rmsnorm(float[] out, float[] x, float[] weight, int size, float eps) {
  float ss = 0f;
  for (int i = 0; i < size; i++) ss += x[i] * x[i];
  ss = 1f / (float) Math.sqrt(ss / size + eps);
  for (int i = 0; i < size; i++) {
  out[i] = weight[i] * (ss * x[i]);
  }
  }
  
  /** SiLU activation: x * sigmoid(x) */
  public static float silu(float x) {
  return x * (1f / (1f + (float) Math.exp(-x)));
  }
  }
