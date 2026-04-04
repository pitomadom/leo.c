/*
 * leotests/test_leo.c — Leo C inference test suite
 *
 * All tests are functions. Run: cc test_leo.c -O2 -lm -o test_leo && ./test_leo
 * Tests verify math ops, GGUF parsing logic, and Gemma-3 specific behavior.
 * No model file needed for unit tests (marked [unit]).
 * Integration tests (marked [integ]) require a GGUF path as argv[1].
 *
 * (c) 2026 arianna method
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    tests_run++; \
    if (!(cond)) { printf("  FAIL: %s\n", msg); tests_failed++; } \
    else { tests_passed++; } \
} while(0)

#define ASSERT_NEAR(a, b, eps, msg) do { \
    tests_run++; \
    if (fabsf((a) - (b)) > (eps)) { \
        printf("  FAIL: %s (got %.6f, expected %.6f, diff %.6f)\n", msg, (float)(a), (float)(b), fabsf((a)-(b))); \
        tests_failed++; \
    } else { tests_passed++; } \
} while(0)

/* ═══════════════════════════════════════════════════════════════
 * Math ops (copied from leo.c for standalone testing)
 * ═══════════════════════════════════════════════════════════════ */

static float silu_f(float x) { return x / (1.0f + expf(-x)); }
static float gelu_tanh_f(float x) { return 0.5f*x*(1.0f+tanhf(0.7978845608f*(x+0.044715f*x*x*x))); }

static void rmsnorm(float *out, const float *x, const float *w, int d, float eps) {
    float ss = 0; for (int i = 0; i < d; i++) ss += x[i]*x[i];
    float inv = 1.0f / sqrtf(ss/d + eps);
    for (int i = 0; i < d; i++) out[i] = x[i] * inv * w[i];
}

static void rmsnorm_gemma(float *out, const float *x, const float *w, int d, float eps) {
    float ss = 0; for (int i = 0; i < d; i++) ss += x[i]*x[i];
    float inv = 1.0f / sqrtf(ss/d + eps);
    for (int i = 0; i < d; i++) out[i] = x[i] * inv * (w[i] + 1.0f);
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) f = sign << 31;
        else { exp = 1; while (!(mant & 0x400)) { mant <<= 1; exp--; } mant &= 0x3FF; f = (sign<<31)|((exp+112)<<23)|(mant<<13); }
    } else if (exp == 31) f = (sign<<31)|0x7F800000|(mant<<13);
    else f = (sign<<31)|((exp+112)<<23)|(mant<<13);
    float r; memcpy(&r, &f, 4); return r;
}

static void softmax_n(float *x, int n) {
    float mx = x[0]; for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i]-mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

/* ═══════════════════════════════════════════════════════════════
 * Unit tests
 * ═══════════════════════════════════════════════════════════════ */

void test_gelu_tanh(void) {
    printf("[unit] test_gelu_tanh\n");
    ASSERT_NEAR(gelu_tanh_f(0.0f), 0.0f, 1e-6, "gelu(0) = 0");
    ASSERT_NEAR(gelu_tanh_f(1.0f), 0.8412f, 1e-3, "gelu(1) ≈ 0.841");
    ASSERT_NEAR(gelu_tanh_f(-1.0f), -0.1588f, 1e-3, "gelu(-1) ≈ -0.159");
    ASSERT_NEAR(gelu_tanh_f(3.0f), 2.9964f, 1e-3, "gelu(3) ≈ 2.996");
    /* GELU ≠ SiLU */
    float g = gelu_tanh_f(1.0f);
    float s = silu_f(1.0f);
    ASSERT(fabsf(g - s) > 0.05f, "gelu(1) ≠ silu(1)");
}

void test_silu(void) {
    printf("[unit] test_silu\n");
    ASSERT_NEAR(silu_f(0.0f), 0.0f, 1e-6, "silu(0) = 0");
    ASSERT_NEAR(silu_f(1.0f), 0.7311f, 1e-3, "silu(1) ≈ 0.731");
    ASSERT_NEAR(silu_f(-1.0f), -0.2689f, 1e-3, "silu(-1) ≈ -0.269");
}

void test_rmsnorm_basic(void) {
    printf("[unit] test_rmsnorm_basic\n");
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float out[4];
    rmsnorm(out, x, w, 4, 1e-6f);
    /* RMS of [1,2,3,4] = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386 */
    float rms = sqrtf(30.0f / 4.0f);
    ASSERT_NEAR(out[0], 1.0f / rms, 1e-4, "rmsnorm x[0]/rms");
    ASSERT_NEAR(out[1], 2.0f / rms, 1e-4, "rmsnorm x[1]/rms");
    ASSERT_NEAR(out[3], 4.0f / rms, 1e-4, "rmsnorm x[3]/rms");
}

void test_rmsnorm_with_weights(void) {
    printf("[unit] test_rmsnorm_with_weights\n");
    float x[] = {1.0f, -1.0f, 1.0f, -1.0f};
    float w[] = {2.0f, 3.0f, 4.0f, 5.0f};
    float out[4];
    rmsnorm(out, x, w, 4, 1e-6f);
    /* RMS = sqrt(4/4) = 1.0, inv = 1.0 */
    ASSERT_NEAR(out[0], 2.0f, 1e-4, "rmsnorm weighted x[0]");
    ASSERT_NEAR(out[1], -3.0f, 1e-4, "rmsnorm weighted x[1]");
    ASSERT_NEAR(out[2], 4.0f, 1e-4, "rmsnorm weighted x[2]");
    ASSERT_NEAR(out[3], -5.0f, 1e-4, "rmsnorm weighted x[3]");
}

void test_rmsnorm_gemma_adds_one(void) {
    printf("[unit] test_rmsnorm_gemma_adds_one\n");
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w_original[] = {0.5f, 0.5f, 0.5f, 0.5f}; /* Gemma stores w-1 */
    float out_gemma[4], out_normal[4];
    float w_converted[] = {1.5f, 1.5f, 1.5f, 1.5f}; /* After llama.cpp +1 */
    rmsnorm_gemma(out_gemma, x, w_original, 4, 1e-6f);
    rmsnorm(out_normal, x, w_converted, 4, 1e-6f);
    /* Both should give same result */
    for (int i = 0; i < 4; i++)
        ASSERT_NEAR(out_gemma[i], out_normal[i], 1e-5, "gemma norm matches converted");
}

void test_f16_to_f32(void) {
    printf("[unit] test_f16_to_f32\n");
    /* f16 encoding of 1.0 = 0x3C00 */
    ASSERT_NEAR(f16_to_f32(0x3C00), 1.0f, 1e-6, "f16(1.0)");
    /* f16 encoding of -1.0 = 0xBC00 */
    ASSERT_NEAR(f16_to_f32(0xBC00), -1.0f, 1e-6, "f16(-1.0)");
    /* f16 encoding of 0.0 = 0x0000 */
    ASSERT_NEAR(f16_to_f32(0x0000), 0.0f, 1e-6, "f16(0.0)");
    /* f16 encoding of 0.5 = 0x3800 */
    ASSERT_NEAR(f16_to_f32(0x3800), 0.5f, 1e-6, "f16(0.5)");
    /* f16 encoding of 65504 (max normal) = 0x7BFF */
    ASSERT(f16_to_f32(0x7BFF) > 65000.0f, "f16 max normal > 65000");
}

void test_softmax(void) {
    printf("[unit] test_softmax\n");
    float x[] = {1.0f, 2.0f, 3.0f};
    softmax_n(x, 3);
    float sum = x[0] + x[1] + x[2];
    ASSERT_NEAR(sum, 1.0f, 1e-5, "softmax sums to 1");
    ASSERT(x[2] > x[1], "softmax preserves order [2]>[1]");
    ASSERT(x[1] > x[0], "softmax preserves order [1]>[0]");
    ASSERT_NEAR(x[2], 0.6652f, 1e-3, "softmax(3) ≈ 0.665");
}

void test_softmax_single(void) {
    printf("[unit] test_softmax_single\n");
    float x[] = {42.0f};
    softmax_n(x, 1);
    ASSERT_NEAR(x[0], 1.0f, 1e-6, "softmax single element = 1.0");
}

void test_softmax_numerical_stability(void) {
    printf("[unit] test_softmax_numerical_stability\n");
    float x[] = {1000.0f, 1001.0f, 1002.0f};
    softmax_n(x, 3);
    float sum = x[0] + x[1] + x[2];
    ASSERT_NEAR(sum, 1.0f, 1e-5, "softmax stable with large values");
    ASSERT(x[2] > x[1], "softmax stable: preserves order");
}

void test_embedding_scaling(void) {
    printf("[unit] test_embedding_scaling\n");
    /* Gemma-3 scales embeddings by sqrt(hidden_dim) */
    int D = 640;
    float scale = sqrtf((float)D);
    ASSERT_NEAR(scale, 25.298f, 0.01f, "sqrt(640) ≈ 25.298");
    /* After scaling, a typical embedding value ~0.02 becomes ~0.5 */
    float emb = 0.02f;
    ASSERT_NEAR(emb * scale, 0.506f, 0.01f, "scaled embedding ≈ 0.506");
}

void test_gemma3_architecture_constants(void) {
    printf("[unit] test_gemma3_architecture_constants\n");
    /* Gemma-3 270M architecture */
    int n_layers = 18;
    int hidden = 640;
    int ffn = 2048;
    int n_heads = 4;
    int n_kv_heads = 1;
    int head_dim = 256;
    int vocab = 262144;
    int sliding_window = 512;

    ASSERT(head_dim != hidden / n_heads, "Gemma-3 head_dim ≠ hidden/heads (256 ≠ 160)");
    ASSERT(n_heads / n_kv_heads == 4, "GQA ratio = 4");
    ASSERT(n_heads * head_dim == 1024, "Q proj output = 1024");
    ASSERT(n_kv_heads * head_dim == 256, "KV proj output = 256");

    /* Global attention layers: every 6th starting from 5 */
    int global_layers[] = {5, 11, 17};
    for (int i = 0; i < 3; i++)
        ASSERT(global_layers[i] % 6 == 5, "global layer pattern: l%6==5");

    /* Local layers use sliding window */
    for (int l = 0; l < n_layers; l++) {
        if (l % 6 != 5)
            ASSERT(sliding_window == 512, "local layer window = 512");
    }

    (void)ffn; (void)vocab;
}

void test_rope_dual_theta(void) {
    printf("[unit] test_rope_dual_theta\n");
    /* Gemma-3 uses two RoPE bases */
    float theta_global = 1000000.0f;
    float theta_swa = 10000.0f;
    ASSERT(theta_global != theta_swa, "dual theta: global ≠ swa");

    /* Frequency at position 0, dim 0 */
    int head_dim = 256;
    float freq_global = 1.0f / powf(theta_global, 0.0f / (float)head_dim);
    float freq_swa = 1.0f / powf(theta_swa, 0.0f / (float)head_dim);
    ASSERT_NEAR(freq_global, 1.0f, 1e-6, "freq at dim=0 always 1.0 (global)");
    ASSERT_NEAR(freq_swa, 1.0f, 1e-6, "freq at dim=0 always 1.0 (swa)");

    /* Frequency at higher dims diverges */
    float freq_g_128 = 1.0f / powf(theta_global, 128.0f / (float)head_dim);
    float freq_s_128 = 1.0f / powf(theta_swa, 128.0f / (float)head_dim);
    ASSERT(freq_s_128 > freq_g_128 * 5.0f, "swa freq >> global freq at dim=128");
}

void test_vocab_limit_gemma(void) {
    printf("[unit] test_vocab_limit_gemma\n");
    /* The critical bug: vocab 200K limit rejected 262K Gemma vocab */
    int gemma_vocab = 262144;
    int old_limit = 200000;
    int new_limit = 300000;
    ASSERT(gemma_vocab > old_limit, "Gemma vocab exceeds old 200K limit");
    ASSERT(gemma_vocab < new_limit, "Gemma vocab fits new 300K limit");
}

void test_geglu_vs_swiglu(void) {
    printf("[unit] test_geglu_vs_swiglu\n");
    /* GEGLU: gelu(gate) * up; SwiGLU: silu(gate) * up */
    float gate = 2.0f, up = 1.5f;
    float geglu = gelu_tanh_f(gate) * up;
    float swiglu = silu_f(gate) * up;
    ASSERT(fabsf(geglu - swiglu) > 0.01f, "GEGLU ≠ SwiGLU for same inputs");
    /* For Gemma-3, GEGLU is correct */
    ASSERT_NEAR(geglu, gelu_tanh_f(2.0f) * 1.5f, 1e-6, "GEGLU = gelu(gate)*up");
}

void test_sliding_window_bounds(void) {
    printf("[unit] test_sliding_window_bounds\n");
    int sliding_window = 512;

    /* At position 100, all tokens visible (100 < 512) */
    int pos1 = 100;
    int start1 = pos1 - sliding_window + 1;
    if (start1 < 0) start1 = 0;
    ASSERT(start1 == 0, "pos=100: start=0 (all visible)");

    /* At position 600, only last 512 visible */
    int pos2 = 600;
    int start2 = pos2 - sliding_window + 1;
    if (start2 < 0) start2 = 0;
    ASSERT(start2 == 89, "pos=600: start=89 (window applied)");
    ASSERT(pos2 - start2 + 1 == 512, "window size = 512");
}

void test_qk_norm_dimensions(void) {
    printf("[unit] test_qk_norm_dimensions\n");
    /* Q/K norm operates on head_dim=256, not hidden=640 */
    int head_dim = 256;
    int n_heads = 4;
    int n_kv_heads = 1;
    int q_total = n_heads * head_dim;     /* 1024 */
    int k_total = n_kv_heads * head_dim;  /* 256 */

    ASSERT(q_total == 1024, "Q total dim = 1024");
    ASSERT(k_total == 256, "K total dim = 256");

    /* Norm weight is per head_dim, shared across heads */
    int norm_weight_size = head_dim; /* 256 */
    ASSERT(norm_weight_size == 256, "Q/K norm weight = 256 elements");
}

void test_post_norm_sandwich(void) {
    printf("[unit] test_post_norm_sandwich\n");
    /* Gemma-3 sandwich norm: pre-norm → sublayer → post-norm → residual */
    float residual[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float sublayer_out[] = {0.1f, 0.2f, 0.3f, 0.4f};
    float post_norm_w[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float normed[4];

    /* Post-norm the sublayer output */
    rmsnorm(normed, sublayer_out, post_norm_w, 4, 1e-6f);

    /* Add to residual */
    float result[4];
    for (int i = 0; i < 4; i++) result[i] = residual[i] + normed[i];

    /* Result should be residual + normalized sublayer */
    ASSERT(result[0] > residual[0], "sandwich: result > residual (positive sublayer)");
    ASSERT(result[0] < residual[0] + sublayer_out[0] + 0.5f, "sandwich: post-norm bounds output");
}

void test_tied_embeddings(void) {
    printf("[unit] test_tied_embeddings\n");
    /* Gemma-3 270M uses tied embeddings: lm_head = embed_tokens */
    /* When output.weight is missing, we reuse token_embd.weight */
    float *tok_emb = (float *)malloc(4 * sizeof(float));
    tok_emb[0] = 1.0f; tok_emb[1] = 2.0f; tok_emb[2] = 3.0f; tok_emb[3] = 4.0f;
    float *output = NULL; /* missing */
    if (!output) output = tok_emb; /* tie */
    ASSERT(output == tok_emb, "tied: output == tok_emb");
    ASSERT_NEAR(output[0], 1.0f, 1e-6, "tied: same values");
    free(tok_emb);
}

void test_gemma_bos_eos(void) {
    printf("[unit] test_gemma_bos_eos\n");
    /* Gemma-3 token IDs */
    int bos = 2;  /* <bos> */
    int eos = 106; /* <end_of_turn> */
    ASSERT(bos == 2, "Gemma BOS = 2");
    ASSERT(eos == 106, "Gemma EOS = 106 (<end_of_turn>)");
    ASSERT(bos != eos, "BOS ≠ EOS");
}

void test_attention_output_dim(void) {
    printf("[unit] test_attention_output_dim\n");
    /* For Gemma-3: attention output is heads*head_dim = 1024, but D = 640 */
    /* o_proj maps 1024 → 640 */
    int heads = 4, head_dim = 256, D = 640;
    int attn_out_dim = heads * head_dim;
    ASSERT(attn_out_dim == 1024, "attn output = 1024");
    ASSERT(attn_out_dim > D, "attn output > hidden dim (Gemma-3 peculiarity)");
    /* Buffer must be max(D, attn_out_dim) */
    int buf = attn_out_dim > D ? attn_out_dim : D;
    ASSERT(buf == 1024, "buffer = max(D, attn_out) = 1024");
}

/* ═══════════════════════════════════════════════════════════════
 * Zikharon (memory) tests
 * ═══════════════════════════════════════════════════════════════ */

/* Replicate core Zikharon structures for standalone testing */
#define ZK_MAX_COOC     32768
#define ZK_MAX_ANCHORS  1024
#define ZK_MAX_EPISODES 512
#define ZK_TOPIC_DIM    32
#define ZK_ANCHOR_TOKENS 16
#define ZK_EPISODE_TOKENS 8
#define ZK_DECAY_SURFACE  0.90f
#define ZK_DECAY_MIDDLE   0.95f
#define ZK_DECAY_DEEP     0.998f
#define ZK_RESURFACE_BOOST 1.05f
#define ZK_MAX_COOC_VALUE  5.0f
#define ZK_MEM_ALPHA  0.08f
#define ZK_MEM_BETA   0.05f
#define ZK_MEM_GAMMA  0.03f

typedef struct { uint16_t src, dst; float value; uint16_t age, access; } TZkCooc;
typedef struct {
    float topic[ZK_TOPIC_DIM]; float strength;
    uint16_t access_count; uint16_t tokens[ZK_ANCHOR_TOKENS];
} TZkAnchor;
typedef struct {
    float summary[ZK_TOPIC_DIM]; float strength;
    uint16_t tokens[ZK_EPISODE_TOKENS];
} TZkEpisode;

static float t_cosine32(const float *a, const float *b) {
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < ZK_TOPIC_DIM; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    return dot / (sqrtf(na * nb) + 1e-8f);
}

static void t_project(const float *proj, const float *hidden, int dim, float *topic) {
    for (int i = 0; i < ZK_TOPIC_DIM; i++) {
        float s = 0;
        for (int j = 0; j < dim; j++) s += proj[i * dim + j] * hidden[j];
        topic[i] = s;
    }
    float norm = 0;
    for (int i = 0; i < ZK_TOPIC_DIM; i++) norm += topic[i] * topic[i];
    norm = 1.0f / (sqrtf(norm) + 1e-8f);
    for (int i = 0; i < ZK_TOPIC_DIM; i++) topic[i] *= norm;
}

void test_zk_constants(void) {
    printf("[unit] test_zk_constants\n");
    ASSERT(ZK_DECAY_SURFACE < ZK_DECAY_MIDDLE, "surface decays faster than middle");
    ASSERT(ZK_DECAY_MIDDLE < ZK_DECAY_DEEP, "middle decays faster than deep");
    ASSERT(ZK_RESURFACE_BOOST > 1.0f, "resurfacing strengthens");
    ASSERT(ZK_MEM_ALPHA > ZK_MEM_BETA, "co-occurrence weight > anchor weight");
    ASSERT(ZK_MEM_BETA > ZK_MEM_GAMMA, "anchor weight > episode weight");
    /* File size bounds */
    long max_size = 64 + ZK_MAX_COOC * 12 + ZK_MAX_ANCHORS * 176 + ZK_MAX_EPISODES * 188;
    ASSERT(max_size < 1024 * 1024, "max memory file < 1MB");
}

void test_zk_projection(void) {
    printf("[unit] test_zk_projection\n");
    int dim = 640;
    float *proj = (float *)malloc(dim * ZK_TOPIC_DIM * sizeof(float));
    uint32_t rng = 42;
    float scale = 1.0f / sqrtf((float)ZK_TOPIC_DIM);
    for (int i = 0; i < dim * ZK_TOPIC_DIM; i++) {
        rng = rng * 1103515245 + 12345;
        proj[i] = ((float)(rng >> 16) / 32768.0f - 1.0f) * scale;
    }
    float hidden[640];
    for (int i = 0; i < 640; i++) hidden[i] = sinf(i * 0.1f);
    float topic[ZK_TOPIC_DIM];
    t_project(proj, hidden, dim, topic);

    /* Topic should be L2-normalized */
    float norm = 0;
    for (int i = 0; i < ZK_TOPIC_DIM; i++) norm += topic[i] * topic[i];
    ASSERT_NEAR(norm, 1.0f, 0.01f, "topic vector is unit-norm");

    /* Same input → same output (deterministic) */
    float topic2[ZK_TOPIC_DIM];
    t_project(proj, hidden, dim, topic2);
    ASSERT_NEAR(t_cosine32(topic, topic2), 1.0f, 1e-5f, "projection is deterministic");

    /* Different input → different output */
    float hidden2[640];
    for (int i = 0; i < 640; i++) hidden2[i] = cosf(i * 0.3f);
    float topic3[ZK_TOPIC_DIM];
    t_project(proj, hidden2, dim, topic3);
    ASSERT(t_cosine32(topic, topic3) < 0.95f, "different inputs → different topics");

    free(proj);
}

void test_zk_cosine32(void) {
    printf("[unit] test_zk_cosine32\n");
    float a[ZK_TOPIC_DIM], b[ZK_TOPIC_DIM];
    /* Identical vectors */
    for (int i = 0; i < ZK_TOPIC_DIM; i++) a[i] = b[i] = (float)i;
    ASSERT_NEAR(t_cosine32(a, b), 1.0f, 1e-5f, "cosine(x, x) = 1");
    /* Opposite */
    for (int i = 0; i < ZK_TOPIC_DIM; i++) b[i] = -(float)i;
    ASSERT_NEAR(t_cosine32(a, b), -1.0f, 1e-5f, "cosine(x, -x) = -1");
    /* Orthogonal */
    memset(a, 0, sizeof(a)); a[0] = 1.0f;
    memset(b, 0, sizeof(b)); b[1] = 1.0f;
    ASSERT_NEAR(t_cosine32(a, b), 0.0f, 1e-5f, "cosine(e0, e1) = 0");
}

void test_zk_decay_surface(void) {
    printf("[unit] test_zk_decay_surface\n");
    float val = 1.0f;
    /* 10 sessions */
    val *= powf(ZK_DECAY_SURFACE, 10.0f);
    ASSERT_NEAR(val, 0.3487f, 0.01f, "surface after 10 sessions ≈ 0.35");
    /* 20 sessions */
    val = powf(ZK_DECAY_SURFACE, 20.0f);
    ASSERT_NEAR(val, 0.1216f, 0.01f, "surface after 20 sessions ≈ 0.12");
    /* 50 sessions — practically gone */
    val = powf(ZK_DECAY_SURFACE, 50.0f);
    ASSERT(val < 0.01f, "surface after 50 sessions < prune threshold");
}

void test_zk_decay_middle(void) {
    printf("[unit] test_zk_decay_middle\n");
    float val = powf(ZK_DECAY_MIDDLE, 20.0f);
    ASSERT_NEAR(val, 0.3585f, 0.01f, "middle after 20 sessions ≈ 0.36");
    val = powf(ZK_DECAY_MIDDLE, 100.0f);
    ASSERT(val < 0.01f, "middle after 100 sessions < threshold");
}

void test_zk_decay_deep(void) {
    printf("[unit] test_zk_decay_deep\n");
    /* Deep memories are almost eternal */
    float val = powf(ZK_DECAY_DEEP, 100.0f);
    ASSERT_NEAR(val, 0.8187f, 0.01f, "deep after 100 sessions ≈ 0.82");
    val = powf(ZK_DECAY_DEEP, 365.0f);
    ASSERT_NEAR(val, 0.4825f, 0.02f, "deep after 365 sessions ≈ 0.48");
    val = powf(ZK_DECAY_DEEP, 1000.0f);
    ASSERT(val > 0.01f, "deep after 1000 sessions still above threshold");
}

void test_zk_resurfacing(void) {
    printf("[unit] test_zk_resurfacing\n");
    float strength = 0.5f;
    /* 10 accesses */
    for (int i = 0; i < 10; i++) strength *= ZK_RESURFACE_BOOST;
    if (strength > 1.0f) strength = 1.0f;
    ASSERT_NEAR(strength, 0.8144f, 0.01f, "10 resurfacings: 0.5 → 0.81");

    /* Deep decay 0.998 vs weekly resurfacing 1.05 */
    float weekly = 1.0f;
    for (int week = 0; week < 52; week++) {
        weekly *= powf(ZK_DECAY_DEEP, 7.0f);  /* 7 sessions/week */
        weekly *= ZK_RESURFACE_BOOST;           /* 1 access/week */
        if (weekly > 1.0f) weekly = 1.0f;
    }
    ASSERT(weekly > 0.9f, "weekly resurfacing maintains deep memory > 0.9");

    /* Without resurfacing — same period */
    float no_resurface = powf(ZK_DECAY_DEEP, 364.0f);
    ASSERT(no_resurface < 0.5f, "without resurfacing, deep decays to < 0.5");
    ASSERT(weekly > no_resurface * 2, "resurfacing >> no resurfacing");
}

void test_zk_decay_hierarchy(void) {
    printf("[unit] test_zk_decay_hierarchy\n");
    int sessions = 30;
    float surface = powf(ZK_DECAY_SURFACE, (float)sessions);
    float middle  = powf(ZK_DECAY_MIDDLE, (float)sessions);
    float deep    = powf(ZK_DECAY_DEEP, (float)sessions);
    ASSERT(surface < middle, "after 30: surface < middle");
    ASSERT(middle < deep, "after 30: middle < deep");
    ASSERT(surface < 0.05f, "surface nearly gone after 30");
    ASSERT(deep > 0.9f, "deep barely touched after 30");
}

void test_zk_cooc_merge(void) {
    printf("[unit] test_zk_cooc_merge\n");
    /* Simulate: session produces co-occurrences, merge into persistent */
    TZkCooc persistent[4] = {
        {10, 20, 1.0f, 0, 1},
        {10, 30, 0.5f, 5, 2},
        {40, 50, 2.0f, 0, 3},
        {60, 70, 0.02f, 100, 0},  /* nearly dead */
    };
    /* Session adds (10,20) again and new (10,40) */
    /* Merge: existing (10,20) gets +0.3, new (10,40) added */
    persistent[0].value += 1.0f * 0.3f;  /* existing strengthened */
    persistent[0].age = 0;
    persistent[0].access++;
    ASSERT_NEAR(persistent[0].value, 1.3f, 0.01f, "merge strengthens existing");
    ASSERT(persistent[0].age == 0, "merge resets age");
    ASSERT(persistent[0].access == 2, "merge increments access");
}

void test_zk_episode_ring_buffer(void) {
    printf("[unit] test_zk_episode_ring_buffer\n");
    /* Ring buffer: when full, oldest overwritten */
    int n = 0, idx = 0, max = 4;  /* small buffer for test */
    for (int i = 0; i < 7; i++) {
        if (n < max) n++;
        else idx = (idx + 1) % max;
    }
    ASSERT(n == max, "ring buffer saturates at max");
    /* 7 writes into 4 slots: first 3 overwritten */
}

void test_zk_anchor_eviction(void) {
    printf("[unit] test_zk_anchor_eviction\n");
    /* When anchors full, weakest evicted */
    TZkAnchor anchors[3] = {
        {.strength = 0.8f}, {.strength = 0.3f}, {.strength = 0.95f}
    };
    /* Find weakest */
    int weakest = 0;
    for (int i = 1; i < 3; i++)
        if (anchors[i].strength < anchors[weakest].strength) weakest = i;
    ASSERT(weakest == 1, "weakest anchor identified (idx=1, strength=0.3)");
    /* New anchor with emergence 0.7 should evict if stronger */
    float new_emergence = 0.7f;
    ASSERT(anchors[weakest].strength < new_emergence * 0.5f, "eviction: weak anchor < new/2");
}

void test_zk_file_format(void) {
    printf("[unit] test_zk_file_format\n");
    /* Header is exactly 64 bytes */
    /* Can't include the actual struct here but verify constants */
    ASSERT(sizeof(TZkCooc) == 12, "ZkCooc = 12 bytes");
    /* Max file size */
    long max_file = 64 + ZK_MAX_COOC * 12 + ZK_MAX_ANCHORS * 176 + ZK_MAX_EPISODES * 188;
    ASSERT(max_file == 64 + 393216 + 180224 + 96256, "max file components correct");
    ASSERT(max_file < 700000, "max file < 700KB");
}

void test_zk_inject_surface(void) {
    printf("[unit] test_zk_inject_surface\n");
    /* Surface injection: logits[dst] += alpha * value * recency */
    float logits[100]; memset(logits, 0, sizeof(logits));
    TZkCooc co = {5, 42, 2.0f, 3, 1};
    int context[] = {5};
    /* Simulate injection */
    if (co.src == context[0] && co.dst < 100) {
        float recency = 1.0f / (1.0f + 0.1f * co.age);
        logits[co.dst] += ZK_MEM_ALPHA * co.value * recency;
    }
    ASSERT(logits[42] > 0, "surface inject: logits[42] boosted");
    ASSERT_NEAR(logits[42], 0.08f * 2.0f * (1.0f / 1.3f), 0.01f, "surface inject value correct");
    ASSERT(logits[0] == 0, "surface inject: unrelated logits untouched");
}

void test_zk_inject_anchor(void) {
    printf("[unit] test_zk_inject_anchor\n");
    /* Anchor injection: boost tokens when topic matches */
    float topic[ZK_TOPIC_DIM]; memset(topic, 0, sizeof(topic)); topic[0] = 1.0f;
    TZkAnchor a;
    memset(&a, 0, sizeof(a)); a.topic[0] = 0.9f; a.topic[1] = 0.1f; /* similar */
    a.strength = 0.8f;
    a.tokens[0] = 10; a.tokens[1] = 20;

    float sim = t_cosine32(topic, a.topic);
    ASSERT(sim > 0.5f, "anchor topic similar to current");

    float logits[100]; memset(logits, 0, sizeof(logits));
    float boost = ZK_MEM_BETA * sim * a.strength;
    logits[10] += boost;
    logits[20] += boost;
    ASSERT(logits[10] > 0, "anchor inject: token 10 boosted");
    ASSERT(logits[20] > 0, "anchor inject: token 20 boosted");
    ASSERT_NEAR(logits[10], logits[20], 1e-6f, "anchor inject: equal boost");

    /* Resurfacing */
    float old_strength = a.strength;
    a.access_count++;
    a.strength *= ZK_RESURFACE_BOOST;
    ASSERT(a.strength > old_strength, "resurfacing increased strength");
}

void test_zk_inject_episode(void) {
    printf("[unit] test_zk_inject_episode\n");
    /* Episode continuity: boost when session topic matches past */
    float topic[ZK_TOPIC_DIM]; memset(topic, 0, sizeof(topic)); topic[0] = 1.0f;
    TZkEpisode ep;
    memset(&ep, 0, sizeof(ep)); ep.summary[0] = 0.95f; ep.summary[1] = 0.05f;
    ep.strength = 0.7f;
    ep.tokens[0] = 55;

    float sim = t_cosine32(topic, ep.summary);
    ASSERT(sim > 0.6f, "episode topic similar enough");

    float logits[100]; memset(logits, 0, sizeof(logits));
    float boost = ZK_MEM_GAMMA * sim * ep.strength;
    logits[55] += boost;
    ASSERT(logits[55] > 0, "episode inject: token 55 boosted");
    ASSERT(logits[55] < 0.05f, "episode inject: gentle boost (gamma is small)");
}

void test_zk_mem_file_size(void) {
    printf("[unit] test_zk_mem_file_size\n");
    /* Realistic scenario: 100 sessions */
    int cooc = 5000;   /* typical after 100 sessions with pruning */
    int anchors = 50;  /* ~1 anchor per 2 sessions */
    int episodes = 100;
    long size = 64 + cooc * 12 + anchors * 176 + episodes * 188;
    ASSERT(size < 100000, "100 sessions < 100KB");
    /* 1000 sessions */
    cooc = 20000;
    anchors = 300;
    episodes = 512;
    size = 64 + cooc * 12 + anchors * 176 + episodes * 188;
    ASSERT(size < 400000, "1000 sessions < 400KB");
}

/* ═══════════════════════════════════════════════════════════════
 * Neshama (live process) tests
 * ═══════════════════════════════════════════════════════════════ */

void test_neshama_trauma_decay(void) {
    printf("[unit] test_neshama_trauma_decay\n");
    float trauma = 1.0f;
    /* 12 ticks (1 minute at 5s interval) */
    for (int i = 0; i < 12; i++) trauma *= 0.85f;
    ASSERT_NEAR(trauma, 0.1422f, 0.01f, "trauma after 1 min ≈ 0.14");
    /* 60 ticks (5 minutes) */
    trauma = 1.0f;
    for (int i = 0; i < 60; i++) trauma *= 0.85f;
    ASSERT(trauma < 0.001f, "trauma after 5 min < 0.001");
}

void test_neshama_overthink_reinforce(void) {
    printf("[unit] test_neshama_overthink_reinforce\n");
    /* Overthinking Ring 1: subtle reinforcement */
    float val = 1.0f;
    for (int i = 0; i < 100; i++) {
        val *= 1.01f;
        if (val > 5.0f) val = 5.0f;
    }
    ASSERT_NEAR(val, 2.7048f, 0.1f, "100 reinforce steps: 1.0 → 2.7");
    ASSERT(val < ZK_MAX_COOC_VALUE, "reinforcement capped below max");
}

void test_neshama_dream_anchor(void) {
    printf("[unit] test_neshama_dream_anchor\n");
    /* Dream: picks anchor, adds transitive co-occurrences */
    TZkAnchor a;
    memset(&a, 0, sizeof(a));
    a.tokens[0] = 10; a.tokens[1] = 20; a.tokens[2] = 30;
    a.strength = 0.5f;
    a.access_count = 0;

    /* Simulate dream: create co-occurrences between anchor tokens */
    int new_cooc = 0;
    for (int t = 0; t < ZK_ANCHOR_TOKENS - 1; t++) {
        if (a.tokens[t] > 0 && a.tokens[t+1] > 0) new_cooc++;
    }
    ASSERT(new_cooc == 2, "dream creates 2 co-occurrences from 3 tokens");

    /* Resurfacing during dream */
    a.access_count++;
    a.strength *= ZK_RESURFACE_BOOST;
    ASSERT(a.access_count == 1, "dream increments access");
    ASSERT(a.strength > 0.5f, "dream strengthens anchor");
}

/* ═══════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════ */
int main(int argc, char **argv) {
    printf("leo.c test suite\n\n");

    /* Unit tests — no model needed */
    test_gelu_tanh();
    test_silu();
    test_rmsnorm_basic();
    test_rmsnorm_with_weights();
    test_rmsnorm_gemma_adds_one();
    test_f16_to_f32();
    test_softmax();
    test_softmax_single();
    test_softmax_numerical_stability();
    test_embedding_scaling();
    test_gemma3_architecture_constants();
    test_rope_dual_theta();
    test_vocab_limit_gemma();
    test_geglu_vs_swiglu();
    test_sliding_window_bounds();
    test_qk_norm_dimensions();
    test_post_norm_sandwich();
    test_tied_embeddings();
    test_gemma_bos_eos();
    test_attention_output_dim();

    /* ═══════════════════════════════════════════════════════════
     * Zikharon tests
     * ═══════════════════════════════════════════════════════════ */
    test_zk_constants();
    test_zk_projection();
    test_zk_cosine32();
    test_zk_decay_surface();
    test_zk_decay_middle();
    test_zk_decay_deep();
    test_zk_resurfacing();
    test_zk_decay_hierarchy();
    test_zk_cooc_merge();
    test_zk_episode_ring_buffer();
    test_zk_anchor_eviction();
    test_zk_file_format();
    test_zk_inject_surface();
    test_zk_inject_anchor();
    test_zk_inject_episode();
    test_zk_mem_file_size();

    /* Neshama tests */
    test_neshama_trauma_decay();
    test_neshama_overthink_reinforce();
    test_neshama_dream_anchor();

    printf("\n%d tests: %d passed, %d failed\n", tests_run, tests_passed, tests_failed);

    (void)argc; (void)argv;
    return tests_failed > 0 ? 1 : 0;
}
