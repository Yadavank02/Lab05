import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from lab05_starter import train_order
from attention import MultiHeadAttention

INK, BODY, MUTED = "#14161B", "#4E5563", "#7C838C"
NAVY, GREEN, AMBER, RULE = "#1C3D6E", "#0E6B57", "#8A6A18", "#D9DCE2"

plt.rcParams.update({
    "figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": RULE, "axes.labelcolor": BODY, "xtick.color": BODY, "ytick.color": BODY,
    "axes.grid": True, "grid.color": RULE, "font.size": 10,
})


def best_of(fn, repeats=3):
    fn()  # warmup call
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def main():
    seeds = (0, 1, 2)
    settings = [
        ("post", 0),
        ("post", 100),
        ("pre", 0),
    ]

    print("12-Block Transformer Stacks (400 updates, lr=3e-3, chance is 0.50)")
    print(f"{'blocks':>6s} {'layout':>6s} {'warmup':>6s}   accuracy for seeds 0, 1, 2   (mean)")
    for norm, warmup in settings:
        accs = [
            train_order(
                "sinusoidal", seed=s, depth=12, norm=norm, warmup=warmup,
                steps=400, lr=3e-3, test_lens=(16,), attention_cls=MultiHeadAttention
            )[0]
            for s in seeds
        ]
        print(f"{12:6d} {norm:>6s} {warmup:6d}   {[round(a, 3) for a in accs]}   ({sum(accs)/3:.3f})")

    print("\nForward + Backward Timing (batch=32, width=32, best of 3 repeats)")
    torch.manual_seed(0)
    d_model, n_heads, batch_size = 32, 4, 32
    lstm = nn.LSTM(d_model, d_model, batch_first=True)
    attn = MultiHeadAttention(d_model, n_heads)
    lengths = [100, 200, 400, 800]

    lstm_ms, attn_ms = [], []
    print(f"{'length':>7s} {'LSTM (ms)':>11s} {'attention (ms)':>16s} {'score matrix (MB)':>19s}")
    for T in lengths:
        xs = torch.randn(batch_size, T, d_model, requires_grad=True)

        def lstm_step():
            lstm.zero_grad()
            if xs.grad is not None:
                xs.grad.zero_()
            out, _ = lstm(xs)
            out.sum().backward()

        def attn_step():
            attn.zero_grad()
            if xs.grad is not None:
                xs.grad.zero_()
            out = attn(xs)
            out.sum().backward()

        t_lstm = best_of(lstm_step, repeats=3) * 1000.0
        t_attn = best_of(attn_step, repeats=3) * 1000.0
        score_mb = batch_size * n_heads * T * T * 4 / 1e6
        lstm_ms.append(t_lstm)
        attn_ms.append(t_attn)
        print(f"{T:7d} {t_lstm:11.2f} {t_attn:16.2f} {score_mb:19.1f}")

    plt.figure(figsize=(6.5, 3.6))
    plt.plot(lengths, lstm_ms, color=AMBER, lw=2, marker="o", label="LSTM (grows about linearly with T)")
    plt.plot(lengths, attn_ms, color=NAVY, lw=2, marker="o", label="Attention (grows about with T^2)")
    plt.xlabel("sequence length (batch of 32)")
    plt.ylabel("forward + backward time (ms)")
    plt.xticks(lengths)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("timing_figure.png", dpi=150)
    plt.close()
    print("\nSaved figure: timing_figure.png")


if __name__ == "__main__":
    main()