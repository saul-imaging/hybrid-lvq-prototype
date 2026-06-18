import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from hybrid_lvq import train_hybrid_lvq, accuracy
from sklearn.datasets import fetch_openml

def load_splice_dataset():
    data = fetch_openml("splice", version=1, as_frame=False)
    X_raw = data.data.astype(str)
    y_raw = data.target

    classes = {c: i for i, c in enumerate(np.unique(y_raw))}
    y = np.array([classes[c] for c in y_raw], dtype=np.int32)

    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0],
               'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    X = np.zeros((X_raw.shape[0], X_raw.shape[1]*4), dtype=np.float64)

    for i, seq in enumerate(X_raw):
        for j, base in enumerate(seq):
            X[i, 4*j:4*j+4] = mapping.get(base, [0, 0, 0, 0])

    return X, y, "DNA-Splice"
    
if __name__ == "__main__":
    np.random.seed(0)

    X, y, name = load_splice_dataset()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=0, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=0, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    nfil, ncol = 6, 10
    total = nfil * ncol

    # === HLQV (librería) ===
    res = train_hybrid_lvq(
        X_train, y_train, X_val, y_val,
        total=total,
        epocas=600,
        # antes: use_levy=True
        schedule="levy",

        alpha0=0.01,
        alpha_min=0.0001,
        alphaerror=0.2,
        n_media_max=50,
        eval_every=1,
        seed_np=0,
        seed_py=7,

        # si quieres conservar “sin early stop” como antes:
        patience=None,      # o un entero si quieres cortar
        verbose_every=100,  # imprime cada 100 épocas como tu script anterior
    )

    W_best = res["W_best"]
    w2 = res["w2"]
    best_acc = res["best_acc"]
    best_epoch = res["best_epoch"]
    htr = res["hist_train"]
    hval = res["hist_val"]

    print("\n====================")
    print(f"Best acc en VAL = {best_acc:.4f} en época {best_epoch}")

    # Accuracy FINAL en TEST
    acc_test = accuracy(W_best, X_test, y_test, w2)
    print(f"Accuracy FINAL en TEST = {acc_test*100:.4f}%")

    # curva
    plt.figure()
    plt.plot(htr, label="Train accuracy")
    plt.plot(hval, label="Validation accuracy")
    plt.axvline(best_epoch - 1, linestyle="--", label=f"Best epoch = {best_epoch}")
    plt.xlabel("Evaluación (épocas)")
    plt.ylabel("Accuracy")
    # plt.ylim(0, 1)
    plt.title("HybridLVQ: Train vs Validation accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()
