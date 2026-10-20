import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(length):
    return torch.tril(torch.ones(length, length, dtype=torch.bool))


def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)

    # Required assertions: each row of weights sums to 1, and masked positions are exactly 0
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-5), (
        "Each row of attention weights must sum to 1."
    )
    if mask is not None:
        assert torch.all(weights.masked_select(~mask) == 0.0), (
            "Masked positions must have attention weight equal to exact 0."
        )

    return weights @ v, weights


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model, self.n_heads, self.d_head = d_model, n_heads, d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch, length, _ = x.shape
        return x.view(batch, length, self.n_heads, self.d_head).transpose(1, 2)  # (batch, heads, T, d_head)

    def forward(self, x, context=None, mask=None):
        context = x if context is None else context
        q = self.split_heads(self.w_q(x))
        k = self.split_heads(self.w_k(context))
        v = self.split_heads(self.w_v(context))
        out, _ = scaled_dot_product_attention(q, k, v, mask=mask)
        batch, _, length, _ = out.shape
        out = out.transpose(1, 2).reshape(batch, length, self.d_model)  # concatenate the heads
        return self.w_o(out)


if __name__ == "__main__":
    torch.manual_seed(0)
    print("Attention From Scratch & Verification")

    # 1. Verify scaled_dot_product_attention against F.scaled_dot_product_attention
    q = torch.randn(2, 4, 5, 8)
    k = torch.randn(2, 4, 7, 8)
    v = torch.randn(2, 4, 7, 8)
    out_unmasked, w_unmasked = scaled_dot_product_attention(q, k, v)
    ref_unmasked = F.scaled_dot_product_attention(q, k, v)
    diff_unmasked = (out_unmasked - ref_unmasked).abs().max().item()
    assert torch.allclose(out_unmasked, ref_unmasked, atol=1e-5)
    print(f"1a. SDPA (unmasked, T_q=5, T_k=7) matches PyTorch: True (max diff = {diff_unmasked:.2e})")

    s = torch.randn(2, 4, 5, 8)
    cmask = causal_mask(5)
    out_causal, w_causal = scaled_dot_product_attention(s, s, s, mask=cmask)
    ref_causal = F.scaled_dot_product_attention(s, s, s, attn_mask=cmask)
    diff_causal = (out_causal - ref_causal).abs().max().item()
    assert torch.allclose(out_causal, ref_causal, atol=1e-5)
    print(f"1b. SDPA (causal mask, T=5)       matches PyTorch: True (max diff = {diff_causal:.2e})")

    # 2. Verify MultiHeadAttention against nn.MultiheadAttention after copying weights
    d_model, n_heads = 32, 4
    mha = MultiHeadAttention(d_model, n_heads)
    ref = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
    with torch.no_grad():
        W, b = ref.in_proj_weight, ref.in_proj_bias
        for lin, i in ((mha.w_q, 0), (mha.w_k, 1), (mha.w_v, 2)):
            lin.weight.copy_(W[i * d_model:(i + 1) * d_model])
            lin.bias.copy_(b[i * d_model:(i + 1) * d_model])
        mha.w_o.weight.copy_(ref.out_proj.weight)
        mha.w_o.bias.copy_(ref.out_proj.bias)

    x = torch.randn(3, 6, d_model)
    context = torch.randn(3, 9, d_model)  # different length for cross-attention

    # Self-attention
    out_mha_self = mha(x)
    ref_mha_self = ref(x, x, x, need_weights=False)[0]
    diff_mha_self = (out_mha_self - ref_mha_self).abs().max().item()
    assert torch.allclose(out_mha_self, ref_mha_self, atol=1e-5)
    print(f"2a. MHA self-attention (T=6)              matches PyTorch: True (max diff = {diff_mha_self:.2e})")

    # Causal self-attention
    mask_6 = causal_mask(6)
    out_mha_causal = mha(x, mask=mask_6)
    ref_mha_causal = ref(x, x, x, attn_mask=~mask_6, need_weights=False)[0]
    diff_mha_causal = (out_mha_causal - ref_mha_causal).abs().max().item()
    assert torch.allclose(out_mha_causal, ref_mha_causal, atol=1e-5)
    print(f"2b. MHA causal self-attention (T=6)       matches PyTorch: True (max diff = {diff_mha_causal:.2e})")

    # Cross-attention (with and without boolean mask on T_q=6, T_k=9)
    cross_mask = torch.ones(6, 9, dtype=torch.bool)
    cross_mask[:, 5:] = False  # block last 4 context positions
    out_mha_cross = mha(x, context=context, mask=cross_mask)
    ref_mha_cross = ref(x, context, context, attn_mask=~cross_mask, need_weights=False)[0]
    diff_mha_cross = (out_mha_cross - ref_mha_cross).abs().max().item()
    assert torch.allclose(out_mha_cross, ref_mha_cross, atol=1e-5)
    print(f"2c. MHA cross-attention (T_q=6, T_k=9)    matches PyTorch: True (max diff = {diff_mha_cross:.2e})")

    # 3. Parameter counts for 1, 2, 4, and 8 heads at d_model=32
    count = lambda m: sum(p.numel() for p in m.parameters())
    counts = {h: count(MultiHeadAttention(d_model, h)) for h in (1, 2, 4, 8)}
    print("\n3. Parameter counts at d_model = 32:")
    for h, c in counts.items():
        print(f"   n_heads = {h}: {c} parameters")
    print(
        "   Explanation: Splitting into heads is a reshape of the same four 32x32 linear layers "
        "(w_q, w_k, w_v, w_o) into n_heads slices of width 32 // n_heads, which adds no parameters."
    )

    # 4. Permutation equivariance: shuffling input shuffles output in the same way
    perm = torch.randperm(x.shape[1])
    shuffled_then_layer = mha(x[:, perm])
    layer_then_shuffled = mha(x)[:, perm]
    diff_perm = (shuffled_then_layer - layer_then_shuffled).abs().max().item()
    assert torch.allclose(layer_then_shuffled, shuffled_then_layer, atol=1e-5)
    print(f"\n4. Permutation check: mha(x[:, perm]) == mha(x)[:, perm]: True (max diff = {diff_perm:.2e})")
