/*
 * leo.c — Language Emergent Organism
 *
 * C inference for Gemma-3 270M + BEST-leo-resonate (merged).
 * Reads standard GGUF files (q4_0, q8_0, f16). Zero external deps.
 *
 * Architecture: 18 layers, hidden=640, ffn=2048, 4 heads (GQA, 1 KV head),
 *               head_dim=256, vocab=262144, sliding window=512
 *
 * Features:
 *   - Own GGUF parser (mmap, zero deps)
 *   - Full Gemma-3 forward pass (RoPE, GQA, SwiGLU, RMS norm)
 *   - DoE parliament (per-layer experts, Hebbian LoRA, birth/death)
 *   - Dario equation overlay (H + F + A on logits)
 *   - Entropy-driven /resonate/ markers
 *   - Leo bootstrap personality
 *
 * Build: cc leo.c -O2 -lm -o leo
 * Usage: ./leo leo-q8_0.gguf
 *
 * Weights: https://huggingface.co/ataeff/g
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <float.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

/* ========================================================================
 * CONFIGURATION
 * ======================================================================== */

#define LEO_VERSION     "2.0.0"
#define MAX_SEQ         2048
#define MAX_GEN         512
#define MAX_LINE        4096
#define MAX_TENSORS     512

/* DoE parliament */
#define DOE_MAX_EXPERTS   16
#define DOE_MIN_EXPERTS   2
#define DOE_RANK          4
#define DOE_VITALITY_DIE  8

/* Dario equation */
#define DARIO_ALPHA  0.15f
#define DARIO_BETA   0.20f
#define DARIO_GAMMA  0.15f

/* Entropy resonance */
#define H_HIGH       0.35f
#define H_LOW        0.12f
#define ENTER_COUNT  3
#define EXIT_COUNT   5

/* ========================================================================
 * GGUF TYPES
 * ======================================================================== */

/* GGML tensor types */
enum {
    GGML_F32  = 0,
    GGML_F16  = 1,
    GGML_Q4_0 = 2,
    GGML_Q4_1 = 3,
    GGML_Q5_0 = 6,
    GGML_Q5_1 = 7,
    GGML_Q8_0 = 8,
    GGML_Q8_1 = 9,
};

/* GGUF metadata value types */
enum {
    GGUF_UINT8 = 0, GGUF_INT8, GGUF_UINT16, GGUF_INT16,
    GGUF_UINT32, GGUF_INT32, GGUF_FLOAT32, GGUF_BOOL,
    GGUF_STRING, GGUF_ARRAY, GGUF_UINT64, GGUF_INT64, GGUF_FLOAT64,
};

/* Q4_0 block: 18 bytes = f16 scale + 16 bytes (32 x 4-bit) */
#define Q4_0_BLOCK   32
#define Q4_0_BYTES   18

/* Q8_0 block: 34 bytes = f16 scale + 32 bytes (32 x int8) */
#define Q8_0_BLOCK   32
#define Q8_0_BYTES   34

/* Tensor info */
typedef struct {
    char name[128];
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t type;
    uint64_t offset;   /* from data section start */
    void *data;        /* pointer into mmap'd region */
    uint64_t n_elements;
} TensorInfo;

/* Model config (from GGUF metadata) */
typedef struct {
    uint32_t n_layers;
    uint32_t hidden;
    uint32_t ffn;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t vocab;
    uint32_t sliding_window;
    uint32_t quant_type;  /* dominant quantization type */
} ModelConfig;

/* Full model */
typedef struct {
    ModelConfig cfg;
    TensorInfo tensors[MAX_TENSORS];
    int n_tensors;

    /* mmap'd file */
    void *mapped;
    size_t mapped_size;
    int fd;
} Model;

/* KV cache */
typedef struct {
    float *k;
    float *v;
    int len;
} KVCache;

/* DoE expert */
typedef struct {
    float *A;  /* [rank * hidden] */
    float *B;  /* [hidden * rank] */
    float vitality;
    int age;
    int low_streak;
    int alive;
} Expert;

typedef struct {
    Expert experts[DOE_MAX_EXPERTS];
    int n_alive;
} Parliament;

typedef struct {
    float *hebbian;
    float *prophecy;
    float *destiny;
    float destiny_mag;
} DarioField;

typedef struct {
    int in_resonance;
    int consecutive_high;
    int consecutive_low;
    int resonance_tokens;
    int total_entries;
    float entropy_sum;
    int total_tokens;
} ResonanceState;

/* ========================================================================
 * UTILITIES
 * ======================================================================== */

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "OOM: %zu\n", n); exit(1); }
    return p;
}

static void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n, sz);
    if (!p) { fprintf(stderr, "OOM: %zu\n", n * sz); exit(1); }
    return p;
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign << 31; }
        else { exp = 1; while (!(mant & 0x400)) { mant <<= 1; exp--; } mant &= 0x3FF; f = (sign<<31)|((exp+112)<<23)|(mant<<13); }
    } else if (exp == 31) { f = (sign<<31)|0x7F800000|(mant<<13); }
    else { f = (sign<<31)|((exp+112)<<23)|(mant<<13); }
    float r; memcpy(&r, &f, 4); return r;
}

/* ========================================================================
 * GGUF PARSER
 * ======================================================================== */

static uint64_t read_u64(const uint8_t **p) { uint64_t v; memcpy(&v, *p, 8); *p += 8; return v; }
static uint32_t read_u32(const uint8_t **p) { uint32_t v; memcpy(&v, *p, 4); *p += 4; return v; }
static void skip_string(const uint8_t **p) { uint64_t len = read_u64(p); *p += len; }
static void read_string(const uint8_t **p, char *buf, int bufsz) {
    uint64_t len = read_u64(p);
    int n = len < (uint64_t)(bufsz-1) ? (int)len : bufsz-1;
    memcpy(buf, *p, n); buf[n] = '\0'; *p += len;
}

/* Skip a GGUF value of given type */
static void skip_gguf_value(const uint8_t **p, uint32_t type) {
    switch (type) {
        case GGUF_UINT8: case GGUF_INT8: case GGUF_BOOL: *p += 1; break;
        case GGUF_UINT16: case GGUF_INT16: *p += 2; break;
        case GGUF_UINT32: case GGUF_INT32: case GGUF_FLOAT32: *p += 4; break;
        case GGUF_UINT64: case GGUF_INT64: case GGUF_FLOAT64: *p += 8; break;
        case GGUF_STRING: skip_string(p); break;
        case GGUF_ARRAY: {
            uint32_t atype = read_u32(p);
            uint64_t count = read_u64(p);
            for (uint64_t i = 0; i < count; i++) skip_gguf_value(p, atype);
            break;
        }
    }
}

static Model *load_gguf(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "Cannot open %s: %s\n", path, strerror(errno)); exit(1); }

    struct stat st;
    fstat(fd, &st);
    size_t file_size = st.st_size;

    void *mapped = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) { fprintf(stderr, "mmap failed\n"); exit(1); }

    const uint8_t *ptr = (const uint8_t *)mapped;
    const uint8_t *file_start = ptr;

    /* Header */
    if (memcmp(ptr, "GGUF", 4) != 0) { fprintf(stderr, "Not a GGUF file\n"); exit(1); }
    ptr += 4;
    uint32_t version = read_u32(&ptr);
    uint64_t n_tensors = read_u64(&ptr);
    uint64_t n_kv = read_u64(&ptr);

    printf("GGUF v%u: %llu tensors, %llu metadata\n", version, (unsigned long long)n_tensors, (unsigned long long)n_kv);

    Model *m = (Model *)xcalloc(1, sizeof(Model));
    m->mapped = mapped;
    m->mapped_size = file_size;
    m->fd = fd;
    m->n_tensors = (int)n_tensors;

    /* Parse metadata — extract model config */
    ModelConfig *cfg = &m->cfg;
    cfg->n_layers = 18;  /* defaults for Gemma-3 270M */
    cfg->hidden = 640;
    cfg->ffn = 2048;
    cfg->n_heads = 4;
    cfg->n_kv_heads = 1;
    cfg->head_dim = 256;
    cfg->vocab = 262144;
    cfg->sliding_window = 512;

    for (uint64_t i = 0; i < n_kv; i++) {
        char key[256];
        read_string(&ptr, key, sizeof(key));
        uint32_t vtype = read_u32(&ptr);

        /* Extract known config values */
        if (strcmp(key, "gemma3_text.block_count") == 0 || strcmp(key, "llama.block_count") == 0) {
            cfg->n_layers = read_u32(&ptr);
        } else if (strcmp(key, "gemma3_text.embedding_length") == 0 || strcmp(key, "llama.embedding_length") == 0) {
            cfg->hidden = read_u32(&ptr);
        } else if (strcmp(key, "gemma3_text.feed_forward_length") == 0 || strcmp(key, "llama.feed_forward_length") == 0) {
            cfg->ffn = read_u32(&ptr);
        } else if (strcmp(key, "gemma3_text.attention.head_count") == 0 || strcmp(key, "llama.attention.head_count") == 0) {
            cfg->n_heads = read_u32(&ptr);
        } else if (strcmp(key, "gemma3_text.attention.head_count_kv") == 0 || strcmp(key, "llama.attention.head_count_kv") == 0) {
            cfg->n_kv_heads = read_u32(&ptr);
        } else if (strcmp(key, "gemma3_text.attention.sliding_window") == 0) {
            cfg->sliding_window = read_u32(&ptr);
        } else {
            skip_gguf_value(&ptr, vtype);
        }
    }
    cfg->head_dim = cfg->hidden / cfg->n_heads;
    /* Gemma-3 uses head_dim=256, but hidden/n_heads = 640/4 = 160.
       Actually head_dim is from config: 256 with n_heads=4 means q_proj is [1024, 640] */
    cfg->head_dim = 256;  /* hardcoded for Gemma-3 */

    printf("Config: %d layers, h=%d, ffn=%d, heads=%d, kv=%d, hd=%d, vocab=%d\n",
           cfg->n_layers, cfg->hidden, cfg->ffn, cfg->n_heads, cfg->n_kv_heads, cfg->head_dim, cfg->vocab);

    /* Parse tensor infos */
    for (uint64_t i = 0; i < n_tensors && i < MAX_TENSORS; i++) {
        TensorInfo *t = &m->tensors[i];
        read_string(&ptr, t->name, sizeof(t->name));
        t->n_dims = read_u32(&ptr);
        t->n_elements = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) {
            t->dims[d] = read_u64(&ptr);
            t->n_elements *= t->dims[d];
        }
        t->type = read_u32(&ptr);
        t->offset = read_u64(&ptr);
    }

    /* Align to data section */
    uint64_t header_size = (uint64_t)(ptr - file_start);
    uint32_t alignment = 32;  /* default GGUF alignment */
    uint64_t data_start = (header_size + alignment - 1) & ~((uint64_t)alignment - 1);
    fprintf(stderr, "DBG header_end=%llu data_start=%llu (expected 6526752)\n",
            (unsigned long long)header_size, (unsigned long long)data_start);

    /* Set data pointers */
    for (int i = 0; i < m->n_tensors; i++) {
        m->tensors[i].data = (void *)(file_start + data_start + m->tensors[i].offset);
        if (i == 0) cfg->quant_type = m->tensors[i].type;
    }

    /* Print first few tensors for verification */
    for (int i = 0; i < 5 && i < m->n_tensors; i++) {
        TensorInfo *t = &m->tensors[i];
        printf("  [%d] %-50s type=%d dims=%llu", i, t->name, t->type, (unsigned long long)t->n_elements);
        for (uint32_t d = 0; d < t->n_dims; d++) printf(" %llu", (unsigned long long)t->dims[d]);
        printf("\n");
    }
    printf("  ... (%d total)\n", m->n_tensors);

    return m;
}

/* Find tensor by name */
static TensorInfo *find_tensor(Model *m, const char *name) {
    for (int i = 0; i < m->n_tensors; i++) {
        if (strcmp(m->tensors[i].name, name) == 0) return &m->tensors[i];
    }
    return NULL;
}

/* ========================================================================
 * DEQUANTIZATION + MATVEC
 * ======================================================================== */

/* Q8_0 matvec with quantized input for precision (like llama.cpp vec_dot_q8_0_q8_0) */
static void matvec_q8_0(float *out, const TensorInfo *w, const float *x, int rows, int cols) {
    const uint8_t *data = (const uint8_t *)w->data;
    int blocks_per_row = cols / Q8_0_BLOCK;

    /* Quantize input x to q8 blocks for precise int accumulation */
    int n_blocks = cols / Q8_0_BLOCK;
    float *x_scales = (float *)xmalloc(n_blocks * sizeof(float));
    int8_t *x_quant = (int8_t *)xmalloc(cols * sizeof(int8_t));

    for (int b = 0; b < n_blocks; b++) {
        float amax = 0.0f;
        int base = b * Q8_0_BLOCK;
        for (int j = 0; j < Q8_0_BLOCK; j++) {
            float av = fabsf(x[base + j]);
            if (av > amax) amax = av;
        }
        float d = amax / 127.0f;
        x_scales[b] = d;
        float id = (d > 0) ? 1.0f / d : 0.0f;
        for (int j = 0; j < Q8_0_BLOCK; j++) {
            int v = (int)roundf(x[base + j] * id);
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            x_quant[base + j] = (int8_t)v;
        }
    }

    for (int r = 0; r < rows; r++) {
        double sum = 0.0;
        const uint8_t *row_data = data + (size_t)r * blocks_per_row * Q8_0_BYTES;
        for (int b = 0; b < blocks_per_row; b++) {
            const uint8_t *block = row_data + b * Q8_0_BYTES;
            uint16_t d_f16; memcpy(&d_f16, block, 2);
            float w_scale = f16_to_f32(d_f16);
            const int8_t *wq = (const int8_t *)(block + 2);
            int base = b * Q8_0_BLOCK;

            /* Integer dot product within block */
            int32_t isum = 0;
            for (int j = 0; j < Q8_0_BLOCK; j++) {
                isum += (int32_t)wq[j] * (int32_t)x_quant[base + j];
            }
            sum += (double)w_scale * (double)x_scales[b] * (double)isum;
        }
        out[r] = (float)sum;
    }

    free(x_scales);
    free(x_quant);
}

/* Dequantize and dot product for Q4_0 — block accumulation for precision */
static void matvec_q4_0(float *out, const TensorInfo *w, const float *x, int rows, int cols) {
    const uint8_t *data = (const uint8_t *)w->data;
    int blocks_per_row = cols / Q4_0_BLOCK;

    for (int r = 0; r < rows; r++) {
        double sum = 0.0;
        const uint8_t *row_data = data + (size_t)r * blocks_per_row * Q4_0_BYTES;
        for (int b = 0; b < blocks_per_row; b++) {
            const uint8_t *block = row_data + b * Q4_0_BYTES;
            uint16_t d_f16; memcpy(&d_f16, block, 2);
            float d = f16_to_f32(d_f16);
            const uint8_t *qs = block + 2;
            int base = b * Q4_0_BLOCK;
            float block_sum = 0.0f;
            for (int j = 0; j < 16; j++) {
                int v0 = (qs[j] & 0x0F) - 8;
                int v1 = (qs[j] >> 4)   - 8;
                block_sum += (float)v0 * x[base + j];
                block_sum += (float)v1 * x[base + j + 16];
            }
            sum += (double)d * (double)block_sum;
        }
        out[r] = (float)sum;
    }
}

/* Dispatch matvec by type */
static void matvec(float *out, const TensorInfo *w, const float *x, int rows, int cols) {
    if (w->type == GGML_Q8_0) matvec_q8_0(out, w, x, rows, cols);
    else if (w->type == GGML_Q4_0) matvec_q4_0(out, w, x, rows, cols);
    else if (w->type == GGML_F16) {
        const uint16_t *d = (const uint16_t *)w->data;
        for (int r = 0; r < rows; r++) {
            float sum = 0;
            for (int c = 0; c < cols; c++) sum += f16_to_f32(d[(size_t)r * cols + c]) * x[c];
            out[r] = sum;
        }
    } else if (w->type == GGML_F32) {
        const float *d = (const float *)w->data;
        for (int r = 0; r < rows; r++) {
            float sum = 0;
            for (int c = 0; c < cols; c++) sum += d[(size_t)r * cols + c] * x[c];
            out[r] = sum;
        }
    } else {
        fprintf(stderr, "Unsupported tensor type: %d\n", w->type);
        exit(1);
    }
}

/* Read f32 vector from tensor (f16 or f32) */
static void read_vec(float *out, const TensorInfo *t, int n) {
    if (t->type == GGML_F32) {
        memcpy(out, t->data, n * sizeof(float));
    } else if (t->type == GGML_F16) {
        const uint16_t *d = (const uint16_t *)t->data;
        for (int i = 0; i < n; i++) out[i] = f16_to_f32(d[i]);
    }
}

/* Embed lookup (f16, f32, or Q8_0) */
static void embed_lookup(float *out, const TensorInfo *t, int token, int dim) {
    if (t->type == GGML_F16) {
        const uint16_t *d = (const uint16_t *)t->data + (size_t)token * dim;
        for (int i = 0; i < dim; i++) out[i] = f16_to_f32(d[i]);
    } else if (t->type == GGML_F32) {
        const float *d = (const float *)t->data + (size_t)token * dim;
        memcpy(out, d, dim * sizeof(float));
    } else if (t->type == GGML_Q8_0) {
        /* Q8_0: each row is dim/32 blocks of 34 bytes each */
        int blocks_per_row = dim / Q8_0_BLOCK;
        size_t row_bytes = (size_t)blocks_per_row * Q8_0_BYTES;
        const uint8_t *row_data = (const uint8_t *)t->data + (size_t)token * row_bytes;
        for (int b = 0; b < blocks_per_row; b++) {
            const uint8_t *block = row_data + b * Q8_0_BYTES;
            uint16_t d_f16; memcpy(&d_f16, block, 2);
            float d = f16_to_f32(d_f16);
            const int8_t *qs = (const int8_t *)(block + 2);
            int base = b * Q8_0_BLOCK;
            for (int j = 0; j < Q8_0_BLOCK; j++) out[base + j] = d * (float)qs[j];
        }
    } else if (t->type == GGML_Q4_0) {
        int blocks_per_row = dim / Q4_0_BLOCK;
        size_t row_bytes = (size_t)blocks_per_row * Q4_0_BYTES;
        const uint8_t *row_data = (const uint8_t *)t->data + (size_t)token * row_bytes;
        for (int b = 0; b < blocks_per_row; b++) {
            const uint8_t *block = row_data + b * Q4_0_BYTES;
            uint16_t d_f16; memcpy(&d_f16, block, 2);
            float d = f16_to_f32(d_f16);
            const uint8_t *qs = block + 2;
            int base = b * Q4_0_BLOCK;
            for (int j = 0; j < 16; j++) {
                out[base + j]      = d * (float)((qs[j] & 0x0F) - 8);
                out[base + j + 16] = d * (float)((qs[j] >> 4)   - 8);
            }
        }
    }
}

/* ========================================================================
 * MATH
 * ======================================================================== */

static void rms_norm(float *out, const float *x, const float *w, int n) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / n + 1e-6f);
    /* GGUF stores norm weights as (1 + original_weight), so just multiply directly */
    for (int i = 0; i < n; i++) out[i] = x[i] * ss * w[i];
}

static float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

static void softmax_inplace(float *x, int n) {
    float max_v = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_v) max_v = x[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max_v); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

/* ========================================================================
 * FORWARD PASS
 * ======================================================================== */

/* Tensor name helpers */
static TensorInfo *layer_tensor(Model *m, int l, const char *suffix) {
    char name[256];
    snprintf(name, sizeof(name), "blk.%d.%s", l, suffix);
    TensorInfo *t = find_tensor(m, name);
    if (!t) { fprintf(stderr, "Missing tensor: %s\n", name); exit(1); }
    return t;
}

static void forward_layer(Model *m, int l, float *x, KVCache *kv, int pos) {
    ModelConfig *c = &m->cfg;
    int H = c->hidden, HD = c->head_dim, NH = c->n_heads, NKV = c->n_kv_heads, FFN = c->ffn;
    int qkv_dim = NH * HD;  /* 4 * 256 = 1024 */

    float *buf = (float *)xmalloc(H * sizeof(float));
    float *q = (float *)xmalloc(qkv_dim * sizeof(float));
    float *k_new = (float *)xmalloc(NKV * HD * sizeof(float));
    float *v_new = (float *)xmalloc(NKV * HD * sizeof(float));
    float *attn_out = (float *)xcalloc(qkv_dim, sizeof(float));

    /* Norms (pre-loaded as f32) */
    float input_ln[640], post_attn_ln[640], pre_ffn_ln[640], post_ffn_ln[640];
    float q_norm[256], k_norm[256];
    read_vec(input_ln, layer_tensor(m, l, "attn_norm.weight"), H);
    read_vec(post_attn_ln, layer_tensor(m, l, "post_attention_norm.weight"), H);
    read_vec(pre_ffn_ln, layer_tensor(m, l, "ffn_norm.weight"), H);
    read_vec(post_ffn_ln, layer_tensor(m, l, "post_ffw_norm.weight"), H);
    read_vec(q_norm, layer_tensor(m, l, "attn_q_norm.weight"), HD);
    read_vec(k_norm, layer_tensor(m, l, "attn_k_norm.weight"), HD);

    /* Pre-attn norm */
    rms_norm(buf, x, input_ln, H);

    if (l == 0) {
        fprintf(stderr, "DBG L0 input[:4]: %.4f %.4f %.4f %.4f\n", x[0], x[1], x[2], x[3]);
        fprintf(stderr, "DBG L0 normed[:4]: %.4f %.4f %.4f %.4f\n", buf[0], buf[1], buf[2], buf[3]);
    }

    /* QKV */
    matvec(q, layer_tensor(m, l, "attn_q.weight"), buf, qkv_dim, H);
    if (l == 0) fprintf(stderr, "DBG L0 q[:4]: %.4f %.4f %.4f %.4f\n", q[0], q[1], q[2], q[3]);
    matvec(k_new, layer_tensor(m, l, "attn_k.weight"), buf, NKV * HD, H);
    matvec(v_new, layer_tensor(m, l, "attn_v.weight"), buf, NKV * HD, H);

    /* Q/K norms (GGUF weights already include +1) */
    for (int h = 0; h < NH; h++) {
        float *qh = q + h * HD;
        float ss = 0;
        for (int i = 0; i < HD; i++) ss += qh[i] * qh[i];
        ss = 1.0f / sqrtf(ss / HD + 1e-6f);
        for (int i = 0; i < HD; i++) qh[i] = qh[i] * ss * q_norm[i];
    }
    for (int h = 0; h < NKV; h++) {
        float *kh = k_new + h * HD;
        float ss = 0;
        for (int i = 0; i < HD; i++) ss += kh[i] * kh[i];
        ss = 1.0f / sqrtf(ss / HD + 1e-6f);
        for (int i = 0; i < HD; i++) kh[i] = kh[i] * ss * k_norm[i];
    }

    /* RoPE */
    float theta = (l % 6 == 5) ? 1000000.0f : 10000.0f;
    for (int h = 0; h < NH; h++) {
        float *qh = q + h * HD;
        for (int i = 0; i < HD; i += 2) {
            float freq = 1.0f / powf(theta, (float)i / HD);
            float angle = pos * freq;
            float c = cosf(angle), s = sinf(angle);
            float q0 = qh[i], q1 = qh[i+1];
            qh[i] = q0*c - q1*s; qh[i+1] = q0*s + q1*c;
        }
    }
    for (int h = 0; h < NKV; h++) {
        float *kh = k_new + h * HD;
        for (int i = 0; i < HD; i += 2) {
            float freq = 1.0f / powf(theta, (float)i / HD);
            float angle = pos * freq;
            float c = cosf(angle), s = sinf(angle);
            float k0 = kh[i], k1 = kh[i+1];
            kh[i] = k0*c - k1*s; kh[i+1] = k0*s + k1*c;
        }
    }

    if (l == 0) {
        fprintf(stderr, "DBG L0 q_normed[:4]: %.4f %.4f %.4f %.4f\n", q[0], q[1], q[2], q[3]);
        fprintf(stderr, "DBG L0 k_normed[:4]: %.4f %.4f %.4f %.4f\n", k_new[0], k_new[1], k_new[2], k_new[3]);
        fprintf(stderr, "DBG L0 q_roped[:4]: %.4f %.4f %.4f %.4f\n", q[0], q[1], q[2], q[3]);
    }

    /* KV cache store */
    size_t kv_off = ((size_t)l * MAX_SEQ + pos) * NKV * HD;
    memcpy(kv->k + kv_off, k_new, NKV * HD * sizeof(float));
    memcpy(kv->v + kv_off, v_new, NKV * HD * sizeof(float));

    /* GQA attention */
    int seq_len = pos + 1;
    int window = c->sliding_window;
    int start = (l % 6 == 5) ? 0 : (pos >= (int)window ? pos - (int)window + 1 : 0);
    float scale = 1.0f / sqrtf((float)HD);

    for (int h = 0; h < NH; h++) {
        float *qh = q + h * HD;
        int kvh = h / (NH / NKV);

        /* For single-token (pos=0), attention is trivial: output = V */
        if (seq_len == 1) {
            float *oh = attn_out + h * HD;
            size_t vt = ((size_t)l * MAX_SEQ + 0) * NKV * HD + kvh * HD;
            for (int i = 0; i < HD; i++) oh[i] = kv->v[vt + i];
        } else {
            float *scores = (float *)xcalloc(seq_len, sizeof(float));
            for (int t = start; t < seq_len; t++) {
                size_t kt = ((size_t)l * MAX_SEQ + t) * NKV * HD + kvh * HD;
                double dot = 0;
                for (int i = 0; i < HD; i++) dot += (double)qh[i] * (double)kv->k[kt + i];
                scores[t] = (float)(dot * (double)scale);
            }
            for (int t = 0; t < start; t++) scores[t] = -1e9f;
            softmax_inplace(scores, seq_len);
            float *oh = attn_out + h * HD;
            for (int t = start; t < seq_len; t++) {
                size_t vt = ((size_t)l * MAX_SEQ + t) * NKV * HD + kvh * HD;
                for (int i = 0; i < HD; i++) oh[i] += scores[t] * kv->v[vt + i];
            }
            free(scores);
        }
    }

    /* O proj + post-attn norm + residual */
    float *oproj = (float *)xmalloc(H * sizeof(float));
    matvec(oproj, layer_tensor(m, l, "attn_output.weight"), attn_out, H, qkv_dim);

    if (l == 0) {
        fprintf(stderr, "DBG L0 oproj[:4]: %.4f %.4f %.4f %.4f\n", oproj[0], oproj[1], oproj[2], oproj[3]);
    }

    rms_norm(buf, oproj, post_attn_ln, H);

    if (l == 0) {
        fprintf(stderr, "DBG L0 post_attn_normed[:4]: %.4f %.4f %.4f %.4f\n", buf[0], buf[1], buf[2], buf[3]);
    }

    for (int i = 0; i < H; i++) x[i] += buf[i];

    if (l == 0) {
        fprintf(stderr, "DBG L0 after attn+res[:4]: %.4f %.4f %.4f %.4f\n", x[0], x[1], x[2], x[3]);
    }

    /* FFN: pre-norm → SwiGLU → post-norm + residual */
    rms_norm(buf, x, pre_ffn_ln, H);
    float *gate = (float *)xmalloc(FFN * sizeof(float));
    float *up = (float *)xmalloc(FFN * sizeof(float));
    matvec(gate, layer_tensor(m, l, "ffn_gate.weight"), buf, FFN, H);
    matvec(up, layer_tensor(m, l, "ffn_up.weight"), buf, FFN, H);
    for (int i = 0; i < FFN; i++) gate[i] = gelu(gate[i]) * up[i];
    float *down = (float *)xmalloc(H * sizeof(float));
    matvec(down, layer_tensor(m, l, "ffn_down.weight"), gate, H, FFN);
    rms_norm(buf, down, post_ffn_ln, H);
    for (int i = 0; i < H; i++) x[i] += buf[i];

    if (l == 0) {
        fprintf(stderr, "DBG L0 after ffn+res[:4]: %.4f %.4f %.4f %.4f\n", x[0], x[1], x[2], x[3]);
    }

    free(buf); free(q); free(k_new); free(v_new);
    free(attn_out); free(oproj); free(gate); free(up); free(down);
}

static void forward(Model *m, int token, KVCache *kv, float *logits) {
    ModelConfig *c = &m->cfg;
    int H = c->hidden, V = c->vocab;

    float *x = (float *)xmalloc(H * sizeof(float));
    TensorInfo *emb = find_tensor(m, "token_embd.weight");
    embed_lookup(x, emb, token, H);
    float scale = sqrtf((float)H);
    for (int i = 0; i < H; i++) x[i] *= scale;

    for (int l = 0; l < (int)c->n_layers; l++) {
        forward_layer(m, l, x, kv, kv->len);
        if (l == 0) {
            fprintf(stderr, "DBG layer0[:8]: ");
            for (int i = 0; i < 8; i++) fprintf(stderr, "%.4f ", x[i]);
            float n = 0; for (int i = 0; i < H; i++) n += x[i]*x[i];
            fprintf(stderr, " norm=%.2f\n", sqrtf(n));
        }
    }

    /* Final norm + logits */
    float final_norm[640];
    read_vec(final_norm, find_tensor(m, "output_norm.weight"), H);
    float *normed = (float *)xmalloc(H * sizeof(float));
    rms_norm(normed, x, final_norm, H);

    /* Logits: normed @ output.weight^T (or token_embd if tied) */
    TensorInfo *lm = find_tensor(m, "output.weight");
    if (!lm) lm = find_tensor(m, "token_embd.weight");
    /* Use matvec for any type — handles Q8_0, Q4_0, F16, F32 */
    matvec(logits, lm, normed, V, H);

    kv->len++;
    free(x); free(normed);
}

/* ========================================================================
 * DOE PARLIAMENT (same as before)
 * ======================================================================== */

static void parliament_init(Parliament *p, int hidden) {
    memset(p, 0, sizeof(Parliament));
    for (int i = 0; i < DOE_MIN_EXPERTS; i++) {
        Expert *e = &p->experts[i];
        e->alive = 1;
        e->vitality = 0.5f;
        e->A = (float *)xmalloc(DOE_RANK * hidden * sizeof(float));
        e->B = (float *)xmalloc(hidden * DOE_RANK * sizeof(float));
        for (int j = 0; j < DOE_RANK * hidden; j++) {
            e->A[j] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
            e->B[j] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
        }
    }
    p->n_alive = DOE_MIN_EXPERTS;
}

static void parliament_lifecycle(Parliament *p) {
    for (int i = 0; i < DOE_MAX_EXPERTS; i++) {
        if (!p->experts[i].alive) continue;
        if (p->experts[i].vitality < 0.1f) {
            p->experts[i].low_streak++;
            if (p->experts[i].low_streak >= DOE_VITALITY_DIE && p->n_alive > DOE_MIN_EXPERTS) {
                p->experts[i].alive = 0; p->n_alive--;
            }
        } else p->experts[i].low_streak = 0;
    }
}

/* ========================================================================
 * DARIO + ENTROPY + SAMPLING (unchanged logic)
 * ======================================================================== */

static void dario_init(DarioField *d, int V, int H) {
    d->hebbian = (float *)xcalloc(V, sizeof(float));
    d->prophecy = (float *)xcalloc(V, sizeof(float));
    d->destiny = (float *)xcalloc(H, sizeof(float));
    d->destiny_mag = 0;
}

static void dario_update(DarioField *d, int token, const float *emb, int H, int V) {
    d->hebbian[token] += 1.0f;
    for (int i = 0; i < V; i++) d->hebbian[i] *= 0.995f;
    d->prophecy[token] = 0;
    for (int i = 0; i < V; i++) d->prophecy[i] *= 0.98f;
    for (int i = 0; i < H; i++) d->destiny[i] = 0.95f * d->destiny[i] + 0.05f * emb[i];
}

static void dario_apply(DarioField *d, float *logits, int V) {
    for (int v = 0; v < V; v++)
        logits[v] += d->hebbian[v] * DARIO_ALPHA + d->prophecy[v] * DARIO_BETA;
}

static float compute_entropy(const float *logits, int n) {
    float max_l = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > max_l) max_l = logits[i];
    float sum = 0;
    float *p = (float *)xmalloc(n * sizeof(float));
    for (int i = 0; i < n; i++) { p[i] = expf(logits[i] - max_l); sum += p[i]; }
    for (int i = 0; i < n; i++) p[i] /= sum;
    float H = 0;
    for (int i = 0; i < n; i++) if (p[i] > 1e-10f) H -= p[i] * log2f(p[i]);
    free(p);
    return H;
}

static int resonance_update(ResonanceState *rs, float h_norm) {
    rs->total_tokens++; rs->entropy_sum += h_norm;
    if (rs->in_resonance) {
        rs->resonance_tokens++;
        if (rs->resonance_tokens > 500) { rs->in_resonance = 0; return -1; }
        if (h_norm < H_LOW) { rs->consecutive_low++; rs->consecutive_high = 0; }
        else rs->consecutive_low = 0;
        if (rs->consecutive_low >= EXIT_COUNT) { rs->in_resonance = 0; rs->resonance_tokens = 0; rs->consecutive_low = 0; return -1; }
    } else {
        if (h_norm > H_HIGH) { rs->consecutive_high++; rs->consecutive_low = 0; }
        else rs->consecutive_high = 0;
        if (rs->consecutive_high >= ENTER_COUNT) { rs->in_resonance = 1; rs->resonance_tokens = 0; rs->consecutive_high = 0; rs->total_entries++; return 1; }
    }
    return 0;
}

static int sample_topk(float *logits, int n, float temp, int topk) {
    if (temp <= 0) { int b = 0; for (int i = 1; i < n; i++) if (logits[i] > logits[b]) b = i; return b; }
    if (topk > 0 && topk < n) {
        float *tmp = (float *)xmalloc(n * sizeof(float));
        memcpy(tmp, logits, n * sizeof(float));
        for (int i = 0; i < topk; i++) { int b = i; for (int j = i+1; j < n; j++) if (tmp[j] > tmp[b]) b = j; float t = tmp[i]; tmp[i] = tmp[b]; tmp[b] = t; }
        float cut = tmp[topk-1]; free(tmp);
        for (int i = 0; i < n; i++) if (logits[i] < cut) logits[i] = -1e9f;
    }
    for (int i = 0; i < n; i++) logits[i] /= temp;
    softmax_inplace(logits, n);
    float r = (float)rand() / RAND_MAX, cs = 0;
    for (int i = 0; i < n; i++) { cs += logits[i]; if (cs >= r) return i; }
    return n - 1;
}

/* ========================================================================
 * BOOTSTRAP
 * ======================================================================== */

static const char *LEO_BOOTSTRAP =
    "LEO is a language engine organism.\n"
    "No weights. No datasets. No internet.\n"
    "Only a small internal seed and whatever you say to it.\n"
    "Pure recursion. Resonant essence.\n"
    "Thunder remembered.\n";

/* ========================================================================
 * GENERATION + MAIN
 * ======================================================================== */

static void generate(Model *m, KVCache *kv, DarioField *dario,
                     Parliament *parlaments, ResonanceState *rs,
                     const int *prompt, int plen, int max_gen, float temp) {
    int V = m->cfg.vocab, H = m->cfg.hidden;
    float *logits = (float *)xmalloc(V * sizeof(float));
    float *emb_buf = (float *)xmalloc(H * sizeof(float));
    float h_max = log2f((float)V);
    int recent[64], n_recent = 0;

    for (int i = 0; i < plen; i++) forward(m, prompt[i], kv, logits);

    int prev = prompt[plen - 1];
    TensorInfo *emb_t = find_tensor(m, "token_embd.weight");

    for (int step = 0; step < max_gen; step++) {
        if (step > 0) forward(m, prev, kv, logits);

        /* DoE + Dario disabled for now — pure forward pass */
        /* for (int l = 0; l < (int)m->cfg.n_layers; l++) parliament_lifecycle(&parlaments[l]); */
        /* embed_lookup(emb_buf, emb_t, prev, H); */
        /* dario_update(dario, prev, emb_buf, H, V); */
        /* dario_apply(dario, logits, V); */

        float ent = compute_entropy(logits, V);
        float h_norm = ent / h_max;
        int event = resonance_update(rs, h_norm);
        if (event == 1) { printf("\n/resonate/\n"); fflush(stdout); }
        else if (event == -1) { printf("\n/resonated/\n"); fflush(stdout); }

        float t = rs->in_resonance ? temp * (1 + 0.3f * h_norm) : temp;
        int topk = rs->in_resonance ? (int)(40 * (1 + 0.3f * h_norm)) : 40;

        for (int i = 0; i < n_recent; i++) {
            if (recent[i] >= 0 && recent[i] < V) {
                if (logits[recent[i]] > 0) logits[recent[i]] /= 1.3f;
                else logits[recent[i]] *= 1.3f;
            }
        }

        float *lc = (float *)xmalloc(V * sizeof(float));
        memcpy(lc, logits, V * sizeof(float));
        int tok = sample_topk(lc, V, t, topk);
        free(lc);

        if (tok == 1 || tok == 106) break;  /* EOS or end_of_turn */

        /* Decode: for now print token ID — proper decode needs vocab from GGUF tokenizer */
        printf("[%d]", tok);
        fflush(stdout);

        recent[n_recent % 64] = tok;
        n_recent++;
        prev = tok;
    }
    printf("\n");
    free(logits); free(emb_buf);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [--prompt TEXT] [--tokens FILE]\n", argv[0]);
        return 1;
    }

    srand(time(NULL));
    printf("Leo v%s — Language Emergent Organism\n", LEO_VERSION);
    printf("Gemma-3 270M + BEST-leo-resonate + DoE + Dario + Entropy\n\n");

    Model *m = load_gguf(argv[1]);
    int V = m->cfg.vocab, H = m->cfg.hidden;

    KVCache kv;
    kv.k = (float *)xcalloc((size_t)m->cfg.n_layers * MAX_SEQ * m->cfg.n_kv_heads * m->cfg.head_dim, sizeof(float));
    kv.v = (float *)xcalloc((size_t)m->cfg.n_layers * MAX_SEQ * m->cfg.n_kv_heads * m->cfg.head_dim, sizeof(float));
    kv.len = 0;

    Parliament *parl = (Parliament *)xcalloc(m->cfg.n_layers, sizeof(Parliament));
    for (int l = 0; l < (int)m->cfg.n_layers; l++) parliament_init(&parl[l], H);

    DarioField dario;
    dario_init(&dario, V, H);
    ResonanceState rs;
    memset(&rs, 0, sizeof(rs));

    /* Parse args */
    char *single_prompt = NULL;
    char *token_file = NULL;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--prompt") == 0 && i+1 < argc) { single_prompt = argv[++i]; }
        else if (strcmp(argv[i], "--tokens") == 0 && i+1 < argc) { token_file = argv[++i]; }
    }

    int debug_logits = 0;
    for (int i = 2; i < argc; i++) if (strcmp(argv[i], "--debug-logits") == 0) debug_logits = 1;

    if (token_file) {
        FILE *tf = fopen(token_file, "rb");
        if (!tf) { fprintf(stderr, "Cannot open %s\n", token_file); return 1; }
        uint32_t nt; if (fread(&nt, 4, 1, tf) < 1) { fclose(tf); return 1; }
        int *toks = (int *)xmalloc(nt * sizeof(int));
        for (uint32_t i = 0; i < nt; i++) { uint32_t t; if (fread(&t, 4, 1, tf) < 1) break; toks[i] = (int)t; }
        fclose(tf);

        if (debug_logits) {
            /* Just compute logits for the last prompt token, print top-10 */
            float *logits = (float *)xmalloc(V * sizeof(float));
            for (int i = 0; i < (int)nt; i++) forward(m, toks[i], &kv, logits);
            /* Find top 10 */
            int top[10]; float topv[10];
            for (int i = 0; i < 10; i++) { top[i] = -1; topv[i] = -1e30f; }
            for (int i = 0; i < V; i++) {
                if (logits[i] > topv[9]) {
                    topv[9] = logits[i]; top[9] = i;
                    for (int j = 8; j >= 0; j--) {
                        if (topv[j+1] > topv[j]) {
                            float tv = topv[j]; topv[j] = topv[j+1]; topv[j+1] = tv;
                            int ti = top[j]; top[j] = top[j+1]; top[j+1] = ti;
                        }
                    }
                }
            }
            printf("C top-10 logits:\n");
            for (int i = 0; i < 10; i++) printf("  %8d (%8.3f)\n", top[i], topv[i]);
            double mean = 0, std = 0;
            for (int i = 0; i < V; i++) mean += logits[i];
            mean /= V;
            for (int i = 0; i < V; i++) std += (logits[i]-mean)*(logits[i]-mean);
            std = sqrt(std / V);
            printf("mean=%.3f std=%.3f\n", mean, std);
            printf("logits[236786]=%.3f\n", logits[236786]);
            free(logits);
        } else {
            float temp = 0.7f;
            for (int ii = 2; ii < argc; ii++) if (strcmp(argv[ii], "--greedy") == 0) temp = 0.0f;
            printf("Loaded %d tokens\nLeo: ", (int)nt);
            generate(m, &kv, &dario, parl, &rs, toks, (int)nt, MAX_GEN, temp);
            printf("  [H̄=%.3f resonance=%d]\n", rs.total_tokens > 0 ? rs.entropy_sum / rs.total_tokens : 0, rs.total_entries);
        }
        free(toks);
    } else {
        printf("Leo ready. Use --tokens FILE with pre-encoded prompts.\n");
        printf("Or run through llama.cpp: llama-cli -m %s\n", argv[1]);
        printf("Thunder remembered.\n\n");

        /* Demo: generate from BOS */
        int bos[] = {2};
        printf("Leo (from BOS): ");
        generate(m, &kv, &dario, parl, &rs, bos, 1, 100, 0.7f);
        printf("  [H̄=%.3f resonance=%d]\n", rs.total_tokens > 0 ? rs.entropy_sum / rs.total_tokens : 0, rs.total_entries);
    }

    munmap(m->mapped, m->mapped_size);
    close(m->fd);
    return 0;
}
