from tinygrad.nn import Linear, GroupNorm
from tinygrad import Tensor, dtypes
import numpy as np

def zero_pad2d(x, padding):
    # padding is a tuple (left, right, top, bottom)
    left, right, top, bottom = padding
    # Create zero tensors for padding
    zero_tensor_top = Tensor.zeros((x.shape[0], top, x.shape[2]))
    zero_tensor_bottom = Tensor.zeros((x.shape[0], bottom, x.shape[2]))
    # Concatenate the zero tensors with the input tensor
    padded_x = Tensor.cat(zero_tensor_top, x, zero_tensor_bottom, dim=1)
    return padded_x

def normalize(x, axis=-1, p=2.0):
    # Compute the norm manually
    norm = (x.abs() ** p).sum(axis=axis, keepdim=True) ** (1 / p)
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

        self.time_shift = zero_pad2d
        self.receptance = Linear(args.n_embd, args.dim_att, bias=False)
        self.key = Linear(args.n_embd, args.dim_att, bias=False)
        self.value = Linear(args.n_embd, args.dim_att, bias=False)
        self.output = Linear(args.dim_att, args.n_embd, bias=False)
        self.ln_x = GroupNorm(self.n_head, args.dim_att, eps=(1e-5) * (args.head_size_divisor ** 2))

    def __call__(self, x, v0):
        B, T, C = x.shape
        H = self.n_head
        xx = self.time_shift(x, (0, 0, 1, 0))[:, 1:, :] - x[:, :-1, :]

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

        w = -Tensor.softplus(-(self.time_decay + Tensor.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2)) - 0.5
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

# Assuming WKV_7_Tinygrad is implemented correctly
class WKV_7_Tinygrad:
    def __init__(self, head_size, ctx_len, dtype):
        self.head_size = head_size
        self.ctx_len = ctx_len
        self.dtype = dtype

    def forward(self, r, w, k, v, kk, a):
        # Implement the forward pass for WKV_7_Tinygrad
        # This is a placeholder implementation
        return r + w + k + v + kk + a

# Test script
def test_rwkv_tmix_x070():
    class Args:
        n_embd = 16
        dim_att = 64
        head_size_a = 8
        head_size_divisor = 1
        ctx_len = 10

    args = Args()
    layer_id = 0

    model = RWKV_Tmix_x070(args, layer_id)
    input_tensor = Tensor.randn(1, 12, args.n_embd)
    v0 = Tensor.zeros(1, 1, args.n_embd)
    output, v0 = model(input_tensor, v0)
    print("Output tensor shape:", output.shape)
    print("Output tensor:", output.numpy())

test_rwkv_tmix_x070()
