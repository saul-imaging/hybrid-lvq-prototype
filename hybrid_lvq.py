import numpy as np
import random
import math

# =============================================================================
# 2) Lévy step + alpha schedule
# =============================================================================
def levy_step(beta=1.5, rng=random):
    sigma_u = (math.gamma(1+beta) * math.sin(math.pi*beta/2) /
               (math.gamma((1+beta)/2)*beta*2**((beta-1)/2))) ** (1/beta)
    u = rng.gauss(0, sigma_u)
    v = rng.gauss(0, 1.0)
    return u / (abs(v) ** (1.0/beta))


def levy_alpha(
    t, T,
    alpha0=0.3,
    alpha_min=0.01,
    p=2.0,
    p_jump=0.12,
    k=0.08,
    beta=1.5,
    freeze_frac=0.7,
    rng=None
):
    if rng is None:
        rng = random

    frac = max(0.0, 1.0 - t / max(1, T))
    alpha = alpha0 * (frac ** p)

    if (t / max(1, T)) < freeze_frac and rng.random() < p_jump:
        jump = abs(levy_step(beta, rng))
        alpha = alpha * (1.0 + k * jump)

    alpha = max(alpha, alpha_min)
    alpha = min(alpha, alpha0)
    return alpha


# =============================================================================
# 3) Inicialización de prototipos
# =============================================================================
def init_prototypes_from_data(X, y, total, seed_np=0):
    rng = np.random.default_rng(seed_np)
    classes = np.unique(y)
    C = len(classes)
    d = X.shape[1]

    por_clase = total // C
    sobrantes = total - por_clase * C

    W_list = []
    w2_list = []

    for i, cls in enumerate(classes):
        m = por_clase + (1 if i < sobrantes else 0)
        idx = np.flatnonzero(y == cls)
        choose = rng.choice(idx, size=m, replace=(idx.size < m))
        W_list.append(X[choose].copy())
        w2_list.append(np.full(m, cls, dtype=np.int32))

    W0 = np.vstack(W_list).reshape(-1, d)
    w2 = np.concatenate(w2_list)
    return W0, w2


# =============================================================================
# 4) Predicción + accuracy
# =============================================================================
def predict_labels(W, X, w2):
    X2 = np.sum(X*X, axis=1, keepdims=True)      # (N,1)
    W2 = np.sum(W*W, axis=1)[None, :]            # (1,M)
    dist2 = X2 + W2 - 2.0*(X @ W.T)              # (N,M)
    winners = np.argmin(dist2, axis=1)
    return w2[winners]


def accuracy(W, X, y, w2):
    y_pred = predict_labels(W, X, w2)
    return float(np.mean(y_pred == y))


# =============================================================================
# 5) Reseed de neuronas "muertas" con fallback
# =============================================================================
def make_reseed_fn(W_ref, w2_ref, X_ref, y_ref, seed_np=0, reseed_noise=0.12):
    rng = np.random.default_rng(seed_np)

    def reseed_neurona(j):
        cls = int(w2_ref[j])

        # bbox de DATOS de esa clase (estable porque X está escalado)
        idx_data = np.flatnonzero(y_ref == cls)
        if idx_data.size >= 2:
            Xc = X_ref[idx_data]
            lo = Xc.min(axis=0)
            hi = Xc.max(axis=0)
            # seguridad numérica
            if np.isfinite(lo).all() and np.isfinite(hi).all() and np.all(hi > lo):
                W_ref[j, :] = rng.uniform(lo, hi)
                return "bbox-data"

        # fallback: punto real + ruido
        if idx_data.size > 0:
            x = X_ref[rng.choice(idx_data)]
            W_ref[j, :] = x + rng.normal(0.0, reseed_noise, size=W_ref.shape[1])
            return "class-rand"

        return "noop"

    return reseed_neurona

# =============================================================================
# Entrenamiento híbrido: online repulsión + batch top-k media
# =============================================================================
def train_hybrid_lvq(
    Xtr, ytr, Xval, yval,
    *,
    total=60,
    epocas=200,

    # alpha schedule
    schedule="levy",           # "levy" | "exp"
    alpha0=0.35,
    alpha_min=0.01,
    decay=0.03,                # solo para "exp"
    levy_p=2.0,
    levy_p_jump=0.12,
    levy_k=0.08,
    levy_beta=1.5,
    levy_freeze_frac=0.7,

    # update rules
    alphaerror=0.25,
    n_media_max=50,

    # dropout
    dropout_p=0.2,
    dropout_warmup=10,
    dropout_stop_frac=0.5,
    ensure_one_per_class=True,

    # reseed
    reseed_noise=0.12,

    # eval / logging
    eval_every=1,
    verbose_every=100,

    # early stopping
    patience=150,              # contador max sin mejorar
    min_delta=0.0,             # mejora mínima para resetear paciencia

    # reproducibilidad
    seed_np=0,
    seed_py=7,

    # extras
    return_last=False,         # si quieres comparar best vs last
):
    """
    Returns dict con:
      - W_best, best_acc, best_epoch
      - hist_train, hist_val
      - w2
      - (opcional) W_last, last_acc, last_epoch
    """
    rng_np = np.random.default_rng(seed_np)
    py_rng = random.Random(seed_py)

    # init prototipos
    W, w2 = init_prototypes_from_data(Xtr, ytr, total=total, seed_np=seed_np)
    reseed_neurona = make_reseed_fn(W, w2, Xtr, ytr, seed_np=seed_np, reseed_noise=reseed_noise)

    classes = np.unique(ytr)
    class_to_idx = {c: np.flatnonzero(w2 == c) for c in classes}

    best_acc = -1.0
    best_W = W.copy()
    best_epoch = 0

    hist_train = []
    hist_val = []

    bad = 0  # contador sin mejorar

    for t in range(epocas):
        # alpha
        if schedule == "levy":
            alpha = levy_alpha(
                t, epocas,
                alpha0=alpha0, alpha_min=alpha_min,
                p=levy_p, p_jump=levy_p_jump, k=levy_k,
                beta=levy_beta, freeze_frac=levy_freeze_frac,
                rng=py_rng
            )
        # elif schedule == "exp":
        #     alpha = exp_alpha(t, alpha0=alpha0, alpha_min=alpha_min, decay=decay)
        else:
            raise ValueError("schedule debe ser 'levy' o 'exp'")

        # asignaciones (aciertos)
        asignaciones = [[] for _ in range(total)]
        idx_perm = rng_np.permutation(Xtr.shape[0])

        # dropout de prototipos
        if (
            t < dropout_warmup
            or dropout_p <= 0.0
            or (t / max(1, epocas)) >= dropout_stop_frac
        ):
            active = np.ones(total, dtype=bool)
        else:
            keep_prob = 1.0 - dropout_p
            active = rng_np.random(total) < keep_prob

            if ensure_one_per_class:
                for c in classes:
                    idx_c = class_to_idx[c]
                    if idx_c.size > 0 and (not np.any(active[idx_c])):
                        active[rng_np.choice(idx_c)] = True

        # ------------------------
        # Online: repulsión si se equivoca (winner masked por dropout)
        # ------------------------
        for ii in idx_perm:
            x = Xtr[ii]
            yi = ytr[ii]

            dist2 = np.sum((W - x) ** 2, axis=1)
            dist2_masked = np.where(active, dist2, np.inf)
            g = int(np.argmin(dist2_masked))

            if w2[g] == yi:
                asignaciones[g].append(ii)
            else:
                W[g] -= (alphaerror * alpha) * (x - W[g])

        # ------------------------
        # Batch: mover cada prototipo hacia la media top-k de aciertos
        # ------------------------
        for j in range(total):
            idxs = asignaciones[j]
            if len(idxs) == 0:
                reseed_neurona(j)
                continue

            Xj = Xtr[idxs]
            wj = W[j]

            dists = np.sum((Xj - wj) ** 2, axis=1)
            k = min(len(dists), int(n_media_max))
            top_k = np.argsort(dists)[:k]
            mu = Xj[top_k].mean(axis=0)

            W[j] = W[j] + alpha * (mu - wj)

        # ------------------------
        # Eval + best + early stop
        # ------------------------
        if (t + 1) % eval_every == 0 or (t == epocas - 1):
            acc_tr = accuracy(W, Xtr, ytr, w2)
            acc_val = accuracy(W, Xval, yval, w2)

            hist_train.append(acc_tr)
            hist_val.append(acc_val)

            # mejora?
            if acc_val > (best_acc + min_delta):
                best_acc = acc_val
                best_W = W.copy()
                best_epoch = t + 1
                bad = 0
            else:
                bad += 1

            if verbose_every and ((t + 1) % verbose_every == 0 or (t == epocas - 1)):
                print(
                    f"época {t+1:4d} | train={acc_tr:.4f} | val={acc_val:.4f} "
                    f"| best_val={best_acc:.4f} (época {best_epoch}) | bad={bad}/{patience}"
                )

            if patience is not None and patience > 0 and bad >= patience:
                break

    out = {
        "name": "HybridLVQ",
        "W_best": best_W,
        "w2": w2,
        "best_acc": float(best_acc),
        "best_epoch": int(best_epoch),
        "hist_train": np.array(hist_train, dtype=float),
        "hist_val": np.array(hist_val, dtype=float),
        "stopped_epoch": int(t + 1),
    }

    if return_last:
        out["W_last"] = W.copy()
        out["last_acc"] = float(accuracy(W, Xval, yval, w2))
        out["last_epoch"] = int(t + 1)

    return out
