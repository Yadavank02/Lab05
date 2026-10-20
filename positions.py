from lab05_starter import train_order
from attention import MultiHeadAttention


def main():
    seeds = (0, 1, 2)

    print("Order Task — Fixed Training Length 16, Tested at (16, 32)")
    print("Chance accuracy is 0.50\n")
    print(f"{'positions':>11s}   accuracy at length 16 / length 32 for seeds 0, 1, 2   (mean 16 / 32)")
    for pos in ("none", "learned", "sinusoidal"):
        runs = [train_order(pos, seed=s, train_len=16, test_lens=(16, 32), attention_cls=MultiHeadAttention) for s in seeds]
        mean_16 = sum(r[0] for r in runs) / 3
        mean_32 = sum(r[1] for r in runs) / 3
        seed_str = "   ".join(f"{r[0]:.3f} / {r[1]:.3f}" for r in runs)
        print(f"{pos:>11s}   {seed_str}   ({mean_16:.3f} / {mean_32:.3f})")

    print("\nOrder Task — Mixed Training Lengths (8, 16, 24), Tested at (16, 32, 48)")
    print("Chance accuracy is 0.50\n")
    print(f"{'positions':>11s}   accuracy at 16 / 32 / 48 for seeds 0, 1, 2   (mean 16 / 32 / 48)")
    for pos in ("learned", "sinusoidal"):
        runs = [
            train_order(
                pos, seed=s, mixed_lens=(8, 16, 24), test_lens=(16, 32, 48), attention_cls=MultiHeadAttention
            )
            for s in seeds
        ]
        m16 = sum(r[0] for r in runs) / 3
        m32 = sum(r[1] for r in runs) / 3
        m48 = sum(r[2] for r in runs) / 3
        seed_str = "   ".join(f"{r[0]:.3f}/{r[1]:.3f}/{r[2]:.3f}" for r in runs)
        print(f"{pos:>11s}   {seed_str}   ({m16:.3f} / {m32:.3f} / {m48:.3f})")


if __name__ == "__main__":
    main()