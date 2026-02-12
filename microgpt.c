/*
 * The most atomic way to train and inference a GPT in pure, dependency-free C.
 * This file is the complete algorithm.
 * Everything else is just efficiency.
 * Adapted from @karpathy's microgpt.py
 *
 * Compile: gcc -O2 -o microgpt microgpt.c -lm
 * Run:     ./microgpt
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* --- Hyperparameters --- */
#define N_EMBD     16
#define N_HEAD     4
#define N_LAYER    1
#define BLOCK_SIZE 8
#define HEAD_DIM   (N_EMBD / N_HEAD)
#define MAX_VOCAB  128
#define MAX_DOCS   40000
#define MAX_DOC_LEN 32
#define NUM_STEPS  500
#define NUM_SAMPLES 20

/* --- Value (autograd scalar) --- */
typedef enum { OP_NONE, OP_ADD, OP_MUL, OP_POW, OP_LOG, OP_EXP, OP_RELU } OpType;

typedef struct Value {
    double data;
    double grad;
    OpType op;
    struct Value *children[2];
    double aux; /* stores exponent for pow */
} Value;

/* Arena allocator for Values */
#define ARENA_SIZE (1 << 26) /* ~67 million Values */
static Value arena[ARENA_SIZE];
static int arena_pos = 0;
static int arena_base = 0; /* reset point for per-step allocations */

static Value *val_new(double data) {
    if (arena_pos >= ARENA_SIZE) {
        fprintf(stderr, "Arena exhausted at %d values\n", arena_pos);
        exit(1);
    }
    Value *v = &arena[arena_pos++];
    v->data = data;
    v->grad = 0;
    v->op = OP_NONE;
    v->children[0] = NULL;
    v->children[1] = NULL;
    v->aux = 0;
    return v;
}

static Value *val_add(Value *a, Value *b) {
    Value *out = val_new(a->data + b->data);
    out->op = OP_ADD;
    out->children[0] = a;
    out->children[1] = b;
    return out;
}

static Value *val_mul(Value *a, Value *b) {
    Value *out = val_new(a->data * b->data);
    out->op = OP_MUL;
    out->children[0] = a;
    out->children[1] = b;
    return out;
}

static Value *val_mul_scalar(Value *a, double s) {
    Value *sv = val_new(s);
    return val_mul(a, sv);
}

static Value *val_add_scalar(Value *a, double s) {
    Value *sv = val_new(s);
    return val_add(a, sv);
}

static Value *val_pow(Value *a, double n) {
    Value *out = val_new(pow(a->data, n));
    out->op = OP_POW;
    out->children[0] = a;
    out->aux = n;
    return out;
}

static Value *val_log(Value *a) {
    Value *out = val_new(log(a->data));
    out->op = OP_LOG;
    out->children[0] = a;
    return out;
}

static Value *val_exp(Value *a) {
    Value *out = val_new(exp(a->data));
    out->op = OP_EXP;
    out->children[0] = a;
    return out;
}

static Value *val_relu(Value *a) {
    Value *out = val_new(a->data < 0 ? 0 : a->data);
    out->op = OP_RELU;
    out->children[0] = a;
    return out;
}

static Value *val_neg(Value *a) { return val_mul_scalar(a, -1.0); }
static Value *val_sub(Value *a, Value *b) { return val_add(a, val_neg(b)); }
static Value *val_div(Value *a, Value *b) { return val_mul(a, val_pow(b, -1.0)); }
static Value *val_sub_scalar(Value *a, double s) { return val_add_scalar(a, -s); }
static Value *val_div_scalar(Value *a, double s) { Value *sv = val_new(s); return val_div(a, sv); }

/* --- Backward pass (topological sort over arena) --- */
static void backward(Value *root) {
    /* All Values from arena_base..arena_pos were created this step.
       They are already in topological order (parents after children)
       because children are always created before the parent that references them. */
    root->grad = 1.0;
    for (int i = arena_pos - 1; i >= arena_base; i--) {
        Value *v = &arena[i];
        Value *a = v->children[0];
        Value *b = v->children[1];
        switch (v->op) {
            case OP_ADD:
                a->grad += v->grad;
                b->grad += v->grad;
                break;
            case OP_MUL:
                a->grad += b->data * v->grad;
                b->grad += a->data * v->grad;
                break;
            case OP_POW:
                a->grad += v->aux * pow(a->data, v->aux - 1.0) * v->grad;
                break;
            case OP_LOG:
                a->grad += (1.0 / a->data) * v->grad;
                break;
            case OP_EXP:
                a->grad += v->data * v->grad;
                break;
            case OP_RELU:
                a->grad += (v->data > 0 ? 1.0 : 0.0) * v->grad;
                break;
            case OP_NONE:
                break;
        }
    }
}

/* --- Seeded PRNG (Mulberry32) --- */
static unsigned int rng_state = 42;

static double rng_uniform(void) {
    rng_state += 0x6D2B79F5;
    unsigned int t = rng_state;
    t = (t ^ (t >> 15)) * (1 | t);
    t = (t + ((t ^ (t >> 7)) * (61 | t))) ^ t;
    return (double)(t ^ (t >> 14)) / 4294967296.0;
}

static double rng_gauss(void) {
    double u, v, s;
    do {
        u = rng_uniform() * 2.0 - 1.0;
        v = rng_uniform() * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    return u * sqrt(-2.0 * log(s) / s);
}

/* --- Shuffle --- */
static void shuffle_docs(int *indices, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = (int)(rng_uniform() * (i + 1));
        int tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }
}

/* --- Parameter storage (persistent across steps) --- */
/* We store all persistent parameters as plain doubles + their gradients. */
/* Layout: wte[vocab][embd], wpe[block][embd], per-layer weights, lm_head[vocab][embd] */

static int vocab_size;
static int n_params;
static double *param_data;
static double *param_grad;
static double *adam_m;
static double *adam_v;

/* Offsets into param arrays */
static int off_wte, off_wpe, off_lm_head;
static int off_attn_wq[N_LAYER], off_attn_wk[N_LAYER], off_attn_wv[N_LAYER];
static int off_attn_wo[N_LAYER], off_mlp_fc1[N_LAYER], off_mlp_fc2[N_LAYER];

static int alloc_matrix(int *offset, int rows, int cols) {
    int o = *offset;
    *offset += rows * cols;
    return o;
}

static void init_params(void) {
    int offset = 0;
    off_wte = alloc_matrix(&offset, vocab_size, N_EMBD);
    off_wpe = alloc_matrix(&offset, BLOCK_SIZE, N_EMBD);
    for (int l = 0; l < N_LAYER; l++) {
        off_attn_wq[l] = alloc_matrix(&offset, N_EMBD, N_EMBD);
        off_attn_wk[l] = alloc_matrix(&offset, N_EMBD, N_EMBD);
        off_attn_wv[l] = alloc_matrix(&offset, N_EMBD, N_EMBD);
        off_attn_wo[l] = alloc_matrix(&offset, N_EMBD, N_EMBD);
        off_mlp_fc1[l] = alloc_matrix(&offset, 4 * N_EMBD, N_EMBD);
        off_mlp_fc2[l] = alloc_matrix(&offset, N_EMBD, 4 * N_EMBD);
    }
    off_lm_head = alloc_matrix(&offset, vocab_size, N_EMBD);
    n_params = offset;

    param_data = (double *)calloc(n_params, sizeof(double));
    param_grad = (double *)calloc(n_params, sizeof(double));
    adam_m = (double *)calloc(n_params, sizeof(double));
    adam_v = (double *)calloc(n_params, sizeof(double));

    /* Initialize with Gaussian noise */
    for (int i = 0; i < n_params; i++) {
        param_data[i] = rng_gauss() * 0.02;
    }
    /* Zero-init for wo and mlp_fc2 (residual projections) */
    for (int l = 0; l < N_LAYER; l++) {
        for (int i = 0; i < N_EMBD * N_EMBD; i++)
            param_data[off_attn_wo[l] + i] = 0.0;
        for (int i = 0; i < N_EMBD * 4 * N_EMBD; i++)
            param_data[off_mlp_fc2[l] + i] = 0.0;
    }
}

/* Wrap persistent params into arena Values for autograd */
static Value **param_values;

static void wrap_params(void) {
    param_values = (Value **)malloc(n_params * sizeof(Value *));
    for (int i = 0; i < n_params; i++) {
        param_values[i] = val_new(param_data[i]);
    }
}

/* After backward, harvest grads from the Value wrappers */
static void harvest_grads(void) {
    for (int i = 0; i < n_params; i++) {
        param_grad[i] = param_values[i]->grad;
    }
}

/* Get a matrix row (as Value*) from wrapped params */
static Value **get_row(int base_off, int row, int cols) {
    return &param_values[base_off + row * cols];
}

/* --- Linear, softmax, rmsnorm on Value* arrays --- */

static void linear_layer(Value **x, int x_len, Value **out, int out_len, int weight_off) {
    for (int i = 0; i < out_len; i++) {
        Value **w_row = get_row(weight_off, i, x_len);
        Value *sum = val_mul(w_row[0], x[0]);
        for (int j = 1; j < x_len; j++) {
            sum = val_add(sum, val_mul(w_row[j], x[j]));
        }
        out[i] = sum;
    }
}

static void softmax_layer(Value **logits, int n, Value **out) {
    double max_val = -1e30;
    for (int i = 0; i < n; i++)
        if (logits[i]->data > max_val) max_val = logits[i]->data;
    Value *exps[MAX_VOCAB];
    Value *total = NULL;
    for (int i = 0; i < n; i++) {
        exps[i] = val_exp(val_sub_scalar(logits[i], max_val));
        total = total ? val_add(total, exps[i]) : exps[i];
    }
    for (int i = 0; i < n; i++)
        out[i] = val_div(exps[i], total);
}

static void rmsnorm_layer(Value **x, int n, Value **out) {
    Value *ms = val_mul(x[0], x[0]);
    for (int i = 1; i < n; i++)
        ms = val_add(ms, val_mul(x[i], x[i]));
    ms = val_div_scalar(ms, (double)n);
    Value *scale = val_pow(val_add_scalar(ms, 1e-5), -0.5);
    for (int i = 0; i < n; i++)
        out[i] = val_mul(x[i], scale);
}

/* --- KV cache --- */
static Value *key_cache[N_LAYER][BLOCK_SIZE][N_EMBD];
static Value *val_cache[N_LAYER][BLOCK_SIZE][N_EMBD];
static int kv_len[N_LAYER];

static void kv_reset(void) {
    for (int l = 0; l < N_LAYER; l++) kv_len[l] = 0;
}

/* --- GPT forward pass --- */
static void gpt_forward(int token_id, int pos_id, Value **logits_out) {
    Value *x[N_EMBD];
    Value *tmp[4 * N_EMBD]; /* scratch */

    /* tok_emb + pos_emb */
    Value **tok_emb = get_row(off_wte, token_id, N_EMBD);
    Value **pos_emb = get_row(off_wpe, pos_id, N_EMBD);
    for (int i = 0; i < N_EMBD; i++)
        x[i] = val_add(tok_emb[i], pos_emb[i]);

    /* initial rmsnorm */
    Value *normed[N_EMBD];
    rmsnorm_layer(x, N_EMBD, normed);
    for (int i = 0; i < N_EMBD; i++) x[i] = normed[i];

    for (int li = 0; li < N_LAYER; li++) {
        /* save residual */
        Value *x_res[N_EMBD];
        for (int i = 0; i < N_EMBD; i++) x_res[i] = x[i];

        /* pre-attention rmsnorm */
        rmsnorm_layer(x, N_EMBD, normed);

        /* Q, K, V projections */
        Value *q[N_EMBD], *k[N_EMBD], *v[N_EMBD];
        linear_layer(normed, N_EMBD, q, N_EMBD, off_attn_wq[li]);
        linear_layer(normed, N_EMBD, k, N_EMBD, off_attn_wk[li]);
        linear_layer(normed, N_EMBD, v, N_EMBD, off_attn_wv[li]);

        /* Store in KV cache */
        int t = kv_len[li];
        for (int i = 0; i < N_EMBD; i++) {
            key_cache[li][t][i] = k[i];
            val_cache[li][t][i] = v[i];
        }
        kv_len[li] = t + 1;

        /* Multi-head attention */
        Value *x_attn[N_EMBD];
        for (int h = 0; h < N_HEAD; h++) {
            int hs = h * HEAD_DIM;

            /* Compute attention logits for this head */
            Value *attn_logits[BLOCK_SIZE];
            for (int ti = 0; ti < kv_len[li]; ti++) {
                Value *dot = val_mul(q[hs], key_cache[li][ti][hs]);
                for (int j = 1; j < HEAD_DIM; j++)
                    dot = val_add(dot, val_mul(q[hs + j], key_cache[li][ti][hs + j]));
                attn_logits[ti] = val_div_scalar(dot, sqrt((double)HEAD_DIM));
            }

            /* Softmax */
            Value *attn_weights[BLOCK_SIZE];
            softmax_layer(attn_logits, kv_len[li], attn_weights);

            /* Weighted sum of values */
            for (int j = 0; j < HEAD_DIM; j++) {
                Value *acc = val_mul(attn_weights[0], val_cache[li][0][hs + j]);
                for (int ti = 1; ti < kv_len[li]; ti++)
                    acc = val_add(acc, val_mul(attn_weights[ti], val_cache[li][ti][hs + j]));
                x_attn[hs + j] = acc;
            }
        }

        /* Output projection + residual */
        linear_layer(x_attn, N_EMBD, x, N_EMBD, off_attn_wo[li]);
        for (int i = 0; i < N_EMBD; i++)
            x[i] = val_add(x[i], x_res[i]);

        /* MLP */
        Value *x_res2[N_EMBD];
        for (int i = 0; i < N_EMBD; i++) x_res2[i] = x[i];

        rmsnorm_layer(x, N_EMBD, normed);
        linear_layer(normed, N_EMBD, tmp, 4 * N_EMBD, off_mlp_fc1[li]);
        for (int i = 0; i < 4 * N_EMBD; i++) {
            Value *r = val_relu(tmp[i]);
            tmp[i] = val_mul(r, r); /* relu^2 */
        }
        linear_layer(tmp, 4 * N_EMBD, x, N_EMBD, off_mlp_fc2[li]);
        for (int i = 0; i < N_EMBD; i++)
            x[i] = val_add(x[i], x_res2[i]);
    }

    /* lm_head */
    linear_layer(x, N_EMBD, logits_out, vocab_size, off_lm_head);
}

/* --- Data --- */
static char docs[MAX_DOCS][MAX_DOC_LEN];
static int n_docs;
static char itos_map[MAX_VOCAB];
static int stoi_map[256]; /* ASCII -> token id */
static int BOS_TOKEN;

static void load_data(void) {
    FILE *f = fopen("input.txt", "r");
    if (!f) {
        fprintf(stderr, "Error: input.txt not found.\n");
        fprintf(stderr, "Please download it first:\n");
        fprintf(stderr, "  curl -o input.txt https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt\n");
        exit(1);
    }
    char line[256];
    n_docs = 0;
    while (fgets(line, sizeof(line), f) && n_docs < MAX_DOCS) {
        /* trim newline */
        int len = (int)strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
            line[--len] = '\0';
        if (len == 0) continue;
        if (len >= MAX_DOC_LEN) len = MAX_DOC_LEN - 1;
        strncpy(docs[n_docs], line, len);
        docs[n_docs][len] = '\0';
        n_docs++;
    }
    fclose(f);

    /* Build vocab: BOS + sorted unique chars */
    int char_seen[256] = {0};
    for (int d = 0; d < n_docs; d++)
        for (int i = 0; docs[d][i]; i++)
            char_seen[(unsigned char)docs[d][i]] = 1;

    memset(stoi_map, -1, sizeof(stoi_map));
    vocab_size = 0;
    itos_map[vocab_size] = 0; /* BOS = index 0, represented as '\0' */
    BOS_TOKEN = 0;
    vocab_size++;

    for (int c = 0; c < 256; c++) {
        if (char_seen[c]) {
            itos_map[vocab_size] = (char)c;
            stoi_map[c] = vocab_size;
            vocab_size++;
        }
    }
    printf("num docs: %d\n", n_docs);
    printf("vocab size: %d\n", vocab_size);
}

/* --- Main --- */
int main(void) {
    load_data();

    /* Shuffle doc indices */
    int *doc_order = (int *)malloc(n_docs * sizeof(int));
    for (int i = 0; i < n_docs; i++) doc_order[i] = i;
    shuffle_docs(doc_order, n_docs);

    init_params();
    printf("num params: %d\n", n_params);

    /* --- Training --- */
    double learning_rate = 1e-2;
    double beta1 = 0.9, beta2 = 0.95, eps_adam = 1e-8;

    for (int step = 0; step < NUM_STEPS; step++) {
        int doc_idx = doc_order[step % n_docs];
        char *doc = docs[doc_idx];
        int doc_len = (int)strlen(doc);

        /* Build token sequence: BOS, chars..., BOS */
        int tokens[BLOCK_SIZE + 2];
        tokens[0] = BOS_TOKEN;
        int seq_len = 1;
        for (int i = 0; i < doc_len && seq_len < BLOCK_SIZE + 1; i++)
            tokens[seq_len++] = stoi_map[(unsigned char)doc[i]];
        tokens[seq_len++] = BOS_TOKEN;

        int n = seq_len - 1;
        if (n > BLOCK_SIZE) n = BLOCK_SIZE;

        /* Reset arena for this step (keep param wrappers fresh each step) */
        arena_pos = 0;
        arena_base = 0;
        wrap_params();
        arena_base = arena_pos; /* backward will go down to param wrappers */
        /* Actually we need backward to reach param wrappers too */
        arena_base = 0;

        kv_reset();

        Value *losses[BLOCK_SIZE];
        for (int pos = 0; pos < n; pos++) {
            Value *logits[MAX_VOCAB];
            gpt_forward(tokens[pos], pos, logits);

            Value *probs[MAX_VOCAB];
            softmax_layer(logits, vocab_size, probs);

            int target = tokens[pos + 1];
            losses[pos] = val_neg(val_log(probs[target]));
        }

        /* Mean loss */
        Value *loss = losses[0];
        for (int i = 1; i < n; i++)
            loss = val_add(loss, losses[i]);
        loss = val_div_scalar(loss, (double)n);

        /* Backward */
        backward(loss);

        /* Harvest gradients from Value wrappers into param_grad */
        harvest_grads();

        /* Adam update */
        double lr_t = learning_rate * (1.0 - (double)step / NUM_STEPS);
        for (int i = 0; i < n_params; i++) {
            double g = param_grad[i];
            adam_m[i] = beta1 * adam_m[i] + (1.0 - beta1) * g;
            adam_v[i] = beta2 * adam_v[i] + (1.0 - beta2) * g * g;
            double m_hat = adam_m[i] / (1.0 - pow(beta1, step + 1));
            double v_hat = adam_v[i] / (1.0 - pow(beta2, step + 1));
            param_data[i] -= lr_t * m_hat / (sqrt(v_hat) + eps_adam);
        }

        printf("step %4d / %4d | loss %.4f\n", step + 1, NUM_STEPS, loss->data);
    }

    /* --- Inference --- */
    double temperature = 0.6;
    printf("\n--- inference ---\n");

    for (int s = 0; s < NUM_SAMPLES; s++) {
        arena_pos = 0;
        arena_base = 0;
        wrap_params();
        kv_reset();

        int token_id = BOS_TOKEN;
        char output[BLOCK_SIZE + 1];
        int out_len = 0;

        for (int pos = 0; pos < BLOCK_SIZE; pos++) {
            Value *logits[MAX_VOCAB];
            gpt_forward(token_id, pos, logits);

            /* Scale by temperature */
            Value *scaled[MAX_VOCAB];
            for (int i = 0; i < vocab_size; i++)
                scaled[i] = val_div_scalar(logits[i], temperature);

            Value *probs[MAX_VOCAB];
            softmax_layer(scaled, vocab_size, probs);

            /* Weighted random sampling */
            double total_w = 0;
            for (int i = 0; i < vocab_size; i++) total_w += probs[i]->data;
            double r = rng_uniform() * total_w;
            token_id = 0;
            for (int i = 0; i < vocab_size; i++) {
                r -= probs[i]->data;
                if (r <= 0) { token_id = i; break; }
            }

            if (token_id == BOS_TOKEN) break;
            if (out_len < BLOCK_SIZE)
                output[out_len++] = itos_map[token_id];
        }
        output[out_len] = '\0';
        printf("sample %d: %s\n", s + 1, output);
    }

    free(param_data);
    free(param_grad);
    free(adam_m);
    free(adam_v);
    free(param_values);
    free(doc_order);
    return 0;
}
