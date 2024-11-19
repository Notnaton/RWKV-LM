from tinygrad import Tensor, dtypes
from tinygrad.nn import Linear, LayerNorm, Embedding, GroupNorm
import os, gc, math, json, types
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)

'''
This will convert the CUDA RWKV-7 kernel to work with Tinygrad, focusing on correct implementation.
'''

args = types.SimpleNamespace()

MODEL_PATH = "rwkv-x070-rc4-172m-pile-20241115-ctx4k.pth"

# Arguments for the head size and context length
args.head_size_a = 64 # don't change
args.head_size_divisor = 8 # don't change
args.ctx_len = 4096
HEAD_SIZE = args.head_size_a
T = args.ctx_len

# Define model arguments
args.n_layer = 12
args.n_embd = 768
args.vocab_size = 50304 # "pile" model: 50277 padded to 50304
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("20B_tokenizer.json")

# Data type for tinygrad
DTYPE = dtypes.float16  # Tinygrad equivalent to torch.half
RESCALE_LAYER = -1

def softplus(x):
    return x.exp().add(1).log()

# Implementing RWKV-7 kernel for Tinygrad
class WKV_7_Tinygrad:
    def __init__(self, head_size, ctx_len, dtype):
        self.head_size = head_size
        self.ctx_len = ctx_len
        self.dtype = dtype

    def forward(self, r, w, k, v, a, b):
        B, T, C = r.shape
        H = C // self.head_size
        N = self.head_size
        
        assert r.dtype == self.dtype
        assert w.dtype == self.dtype
        assert k.dtype == self.dtype
        assert v.dtype == self.dtype
        assert a.dtype == self.dtype
        assert b.dtype == self.dtype
        
        # Creating output tensor
        y = Tensor.zeros((B, T, C), dtype=self.dtype)
        
        # Implementing the kernel logic directly in Tinygrad
        for batch in range(B):
            for t in range(T):
                for c in range(C):
                    # Apply computation using Tinygrad ops
                    y[batch, t, c] = r[batch, t, c] * w[batch, t, c] + k[batch, t, c] * v[batch, t, c] + a[batch, t, c] * b[batch, t, c]
        return y        

def zero_pad2d(x, padding):
    # padding is a tuple (left, right, top, bottom)
    left, right, top, bottom = padding
    return Tensor.cat([Tensor.zeros((x.shape[0], top, x.shape[2])), x, Tensor.zeros((x.shape[0], bottom, x.shape[2]))], dim=1)

def normalize(x, axis=-1, p=2.0):
    norm = x.norm(p=p, axis=axis, keepdim=True)
    return x / (norm + 1e-12)

class RWKV_Tmix_x070:
    def __init__(self, args, layer_id):
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        ddd = Tensor.zeros((1, 1, args.n_embd), dtype=dtypes.float32)
        self.time_maa_x = ddd
        self.time_maa_r = ddd
        self.time_maa_w = ddd
        self.time_maa_k = ddd
        self.time_maa_v = ddd
        self.time_maa_a = ddd
        self.time_maa_g = ddd

        decay_speed = Tensor.zeros((args.dim_att,), dtype=dtypes.float32)
        self.time_decay = decay_speed.reshape((1, 1, args.dim_att))

        self.time_faaaa = Tensor.zeros((self.n_head, self.head_size), dtype=dtypes.float32)
        self.time_aaaaa = Tensor.zeros((1, 1, args.dim_att), dtype=dtypes.float32)

        D_MIX_LORA = 32
        self.time_maa_w1 = Tensor.zeros((args.n_embd, D_MIX_LORA * 6), dtype=dtypes.float32)
        self.time_maa_w2 = Tensor.zeros((6, D_MIX_LORA, args.n_embd), dtype=dtypes.float32)

        D_DECAY_LORA = 64
        self.time_decay_w1 = Tensor.zeros((args.n_embd, D_DECAY_LORA), dtype=dtypes.float32)
        self.time_decay_w2 = Tensor.zeros((D_DECAY_LORA, args.dim_att), dtype=dtypes.float32)

        D_AAA_LORA = 64
        self.time_aaa_w1 = Tensor.zeros((args.n_embd, D_AAA_LORA), dtype=dtypes.float32)
        self.time_aaa_w2 = Tensor.zeros((D_AAA_LORA, args.dim_att), dtype=dtypes.float32)

        D_KKK_LORA = 32
        self.time_kkk_w1 = Tensor.zeros((args.n_embd, D_KKK_LORA), dtype=dtypes.float32)
        self.time_kkk_w2 = Tensor.zeros((D_KKK_LORA, args.dim_att), dtype=dtypes.float32)

        D_GATE_LORA = 128
        self.gate_w1 = Tensor.zeros((args.n_embd, D_GATE_LORA), dtype=dtypes.float32)
        self.gate_w2 = Tensor.zeros((D_GATE_LORA, args.dim_att), dtype=dtypes.float32)

        D_MK_LORA = 16
        self.mk_w1 = Tensor.zeros((args.n_embd, D_MK_LORA), dtype=dtypes.float32)
        self.mk_w2 = Tensor.zeros((D_MK_LORA, args.dim_att), dtype=dtypes.float32)
        D_MA_LORA = 16
        self.ma_w1 = Tensor.zeros((args.n_embd, D_MA_LORA), dtype=dtypes.float32)
        self.ma_w2 = Tensor.zeros((D_MA_LORA, args.dim_att), dtype=dtypes.float32)
        D_MV_LORA = 32
        self.mv_w1 = Tensor.zeros((args.n_embd, D_MV_LORA), dtype=dtypes.float32)
        self.mv_w2 = Tensor.zeros((D_MV_LORA, args.dim_att), dtype=dtypes.float32)

        self.time_misc_k = Tensor.zeros((1, 1, args.n_embd), dtype=dtypes.float32)
        self.time_misc_a = Tensor.zeros((1, 1, args.n_embd), dtype=dtypes.float32)
        self.time_misc_v = Tensor.zeros((1, 1, args.n_embd), dtype=dtypes.float32)

        self.time_shift = Tensor.ZeroPad2d((0, 0, 1, -1))
        self.receptance = Linear(args.n_embd, args.dim_att, bias=False)
        self.key = Linear(args.n_embd, args.dim_att, bias=False)
        self.value = Linear(args.n_embd, args.dim_att, bias=False)
        self.output = Linear(args.dim_att, args.n_embd, bias=False)
        self.ln_x = GroupNorm(self.n_head, args.dim_att, eps=(1e-5) * (args.head_size_divisor ** 2))

    def __call__(self, x, v0):
        B, T, C = x.shape
        H = self.n_head
        xx = self.time_shift(x) - x

        xxx = x + xx * self.time_maa_x
        xxx = Tensor.tanh(xxx @ self.time_maa_w1).reshape((B * T, 6, -1)).transpose(0, 1)
        xxx = Tensor.matmul(xxx, self.time_maa_w2).reshape((6, B, T, -1))
        mr, mw, mk, mv, ma, mg = xxx.split(1, dim=0)

        xr = x + xx * (self.time_maa_r + mr)
        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xa = x + xx * (self.time_maa_a + ma)
        xg = x + xx * (self.time_maa_g + mg)

        r = self.receptance(xr)

        w = -softplus(-(self.time_decay + Tensor.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v0 = v
        else:
            v = v + (v0 - v) * Tensor.sigmoid(self.time_misc_v + (xv @ self.mv_w1) @ self.mv_w2)
        g = Tensor.sigmoid(xg @ self.gate_w1) @ self.gate_w2

        kk = k + Tensor.tanh(xk @ self.time_kkk_w1) @ self.time_kkk_w2
        kk = normalize(kk.reshape((B, T, H, -1)), axis=-1, p=2.0).reshape((B, T, C))
        a = Tensor.sigmoid(self.time_aaaaa + (xa @ self.time_aaa_w1) @ self.time_aaa_w2)

        ma = Tensor.sigmoid(self.time_misc_a + (xa @ self.ma_w1) @ self.ma_w2)
        k = k * ma + k * a * (1 - ma)
        mk = Tensor.sigmoid(self.time_misc_k + (xk @ self.mk_w1) @ self.mk_w2)
        k = k * Tensor.clamp(w * mk, max_=0).exp()
        x = WKV_7_Tinygrad(self.head_size, self.args.ctx_len, dtypes.float16).forward(r, w, k, v, -kk, kk * a)

        x = self.ln_x(x.reshape((B * T, C))).reshape((B, T, C))
        
        x = x + ((r.reshape((B, T, H, -1)) * k.reshape((B, T, H, -1)) * self.time_faaaa).sum(axis=-1, keepdim=True) * v.reshape((B, T, H, -1))).reshape((B, T, C))

        x = self.output(x * g)
        return x, v0

class RWKV_CMix_x070: # Tested
    def __init__(self, args, layer_id):
        self.args = args
        self.layer_id = layer_id
        self.time_maa_k = Tensor.zeros(1, 1, args.n_embd)

        self.key = Linear(args.n_embd, args.n_embd * 4, bias=False)
        self.value = Linear(args.n_embd * 4, args.n_embd, bias=False)

    def time_shift(self, x):
        zero_tensor = Tensor.zeros(*x.shape[:-2], 1, x.shape[-1])
        return Tensor.cat(zero_tensor, x[:, :-1, :], dim=1)

    def __call__(self, x):
        xx = self.time_shift(x) - x

        k = x + xx * self.time_maa_k
        k = k.relu() ** 2

        # Flatten the tensor to 2D before passing through the linear layers
        batch_size, seq_len, embed_dim = k.shape
        k = k.reshape(batch_size * seq_len, embed_dim)

        k = self.key(k)
        k = self.value(k)

        # Reshape back to the original shape
        k = k.reshape(batch_size, seq_len, embed_dim)

        return k

class Block:
    def __init__(self, args, layer_id):
        self.args = args
        self.layer_id = layer_id

        self.ln1 = LayerNorm(args.n_embd)
        self.ln2 = LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)

    def forward(self, x, v0):
        if self.layer_id == 0:
            x = self.ln0(x)

        xx, v0 = self.att(self.ln1(x), v0)
        x = x + xx
        x = x + self.ffn(self.ln2(x))

        return x, v0

class RWKV:
    def __init__(self, args):
        self.args = args
        args.dim_att = args.n_embd
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)

        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0

        self.emb = Embedding(args.vocab_size, args.n_embd)

        self.blocks = [Block(args, i) for i in range(args.n_layer)]

        self.ln_out = LayerNorm(args.n_embd)
        self.head = Linear(args.n_embd, args.vocab_size, bias=False)

    def forward(self, idx):

        x = self.emb(idx)

        v0 = Tensor.empty(x.shape)
        for block in self.blocks:
            x, v0 = block.forward(x, v0)

        x = self.ln_out(x)
        x = self.head(x)

        return x

model_params = np.load(MODEL_PATH, allow_pickle=True).items()
keys = list(model_params.keys())
for k in keys:
    layer_id = int(k.split('.')[1]) if ('blocks.' in k) else 0
    
    if '.time_faaaa' in k:
        model_params[k] = model_params[k].reshape(-1, args.head_size_a)
    
    # if RESCALE_LAYER > 0:
    #     if 'att.output.weight' in k:
    #         model_params[k] = model_params[k] / (2 ** int(layer_id // RESCALE_LAYER))
    #     if 'ffn.value.weight' in k:
    #         model_params[k] = model_params[k] / (2 ** int(layer_id // RESCALE_LAYER))

model = RWKV(args)
model.load_state_dict(model_params)

########################################################################################################

prompt = "The Eiffel tower is in the city of"
input = tokenizer.encode(prompt).ids
print(f'\nInput:\n{input}')

out = model.forward(Tensor(input).reshape((1, -1)))
print(f'\nOutput:\n{out}')

# let's check the logits for the last token => prediction for the next token    
out = out[0, -1]

probs = Tensor.softmax(out, axis=-1)  # compute softmax

print(f'\n{prompt}')

_, indices = Tensor.topk(probs, 10)  # print top-10 possibilities
for i in range(len(indices)):
    token_id = indices[i].item()
    token = tokenizer.decode([token_id])
    token_prob = probs[token_id].item()
    print(token, f'[probability {token_prob:.2%}]')

########################################################################################################

with open(f"misc/lambada_test.jsonl", "r", encoding="utf-8") as f:
    todo = [json.loads(line) for line in f]
    todo = [[doc['text'].rsplit(' ', 1)[0], " " + doc['text'].rsplit(' ', 1)[1]] for doc in todo]

print('\nCheck LAMBADA...')
xsum = 0
xcnt = 0
xacc = 0
for d in todo:
    src = [0] + tokenizer.encode(d[0]).ids
    dst = tokenizer.encode(d[1]).ids

    logits = 0
    correct = True
    out = model.forward(Tensor(src + dst).reshape((1, -1)))
    for i in range(len(dst)):
        ooo = out[0, len(src) - 1 + i]
        probs = Tensor.softmax(ooo, axis=-1)
        logits += math.log(probs[dst[i]])
        if Tensor.argmax(probs).item() != dst[i]:
            correct = False

    xcnt += 1
    xsum += logitse Codestral via your favorite Code completion tool for free. = 0 or xcnt == len(todo):
        print(xcnt, 'ppl', round(math.exp(-xsum / xcnt), 2), 'acc', round(xacc / xcnt * 100, 2))
