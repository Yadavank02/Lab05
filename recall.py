import matplotlib.pyplot as plt
from lab05_starter import train_recall
from attention import scaled_dot_product_attention

INK, BODY, MUTED = "#14161B", "#4E5563", "#7C838C"
NAVY, GREEN, AMBER, RULE = "#1C3D6E", "#0E6B57", "#8A6A18", "#D9DCE2"

plt.rcParams.update({
    "figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": RULE, "axes.labelcolor": BODY, "xtick.color": BODY, "ytick.color": BODY,
    "axes.grid": True, "grid.color": RULE, "font.size": 10,
})


def main():
    lengths = (30, 100, 400)
    seeds = (0, 1, 2)

    print("=== Piece 2: Delayed Recall — Attention vs. LSTM (chance is 0.25) ===")
    attn_results = {}
    lstm_results = {}

    for T in lengths:
        attn_runs = [train_recall("attention", seq_len=T, seed=s, attention_fn=scaled_dot_product_attention) for s in seeds]
        lstm_runs = [train_recall("lstm", seq_len=T, seed=s) for s in seeds]
        attn_results[T] = attn_runs
        lstm_results[T] = lstm_runs

        attn_accs = [a for a, _ in attn_runs]
        attn_w0s = [w for _, w in attn_runs]
        lstm_accs = [a for a, _ in lstm_runs]

        print(f"\nSequence length T = {T}:")
        print(f"  Attention accuracy (seeds 0, 1, 2): {[round(a, 3) for a in attn_accs]}  (mean = {sum(attn_accs)/3:.3f})")
        print(f"  Attention weight on step 0:         {[round(w, 4) for w in attn_w0s]}  (mean = {sum(attn_w0s)/3:.4f}, uniform 1/T = {1/T:.4f})")
        print(f"  LSTM accuracy      (seeds 0, 1, 2): {[round(a, 3) for a in lstm_accs]}  (mean = {sum(lstm_accs)/3:.3f})")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    x_vals = list(lengths)

    attn_mean_acc = [sum(a for a, _ in attn_results[T]) / 3 for T in lengths]
    lstm_mean_acc = [sum(a for a, _ in lstm_results[T]) / 3 for T in lengths]
    attn_mean_w0 = [sum(w for _, w in attn_results[T]) / 3 for T in lengths]

    for s in seeds:
        axes[0].scatter(x_vals, [attn_results[T][s][0] for T in lengths], color=GREEN, alpha=0.5, zorder=3)
        axes[0].scatter(x_vals, [lstm_results[T][s][0] for T in lengths], color=AMBER, alpha=0.5, zorder=3)

    axes[0].plot(x_vals, attn_mean_acc, marker="o", lw=2, color=GREEN, label="Attention readout (mean)")
    axes[0].plot(x_vals, lstm_mean_acc, marker="o", lw=2, color=AMBER, label="LSTM, forget bias 3 (mean)")
    axes[0].axhline(0.25, color=MUTED, ls="--", lw=1, label="chance (0.25)")
    axes[0].set_xlabel("sequence length T")
    axes[0].set_ylabel("validation accuracy")
    axes[0].set_xticks(x_vals)
    axes[0].set_ylim(0.15, 1.05)
    axes[0].set_title("Recall accuracy vs. sequence length", color=INK, fontsize=10)
    axes[0].legend(frameon=False, loc="center left")

    axes[1].plot(x_vals, attn_mean_w0, marker="o", lw=2, color=NAVY, label="Learned weight on step 0")
    axes[1].plot(x_vals, [1 / T for T in lengths], marker="s", lw=1.5, ls="--", color=MUTED, label="Uniform weight (1/T)")
    axes[1].set_xlabel("sequence length T")
    axes[1].set_ylabel("attention weight on step 0")
    axes[1].set_xticks(x_vals)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("Attention weight placed on step 0", color=INK, fontsize=10)
    axes[1].legend(frameon=False, loc="center right")

    plt.tight_layout()
    plt.savefig("recall_figure.png", dpi=150)
    plt.close()
    print("\nSaved figure: recall_figure.png")


if __name__ == "__main__":
    main()