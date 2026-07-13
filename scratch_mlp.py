import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

BS = 32
LR = 1e-3
MAX_ITER = 20000
SPLIT_RATIO = 0.9
WINDOW = 200

def data_preproc(path: str):
    df = pd.read_csv(path, sep="\t")
    if df.isna().values.any():
        logging.warning("na vals detected, dropping them")
        df.dropna(axis=0, inplace=True)
    df = df.iloc[np.random.permutation(len(df))].reset_index(drop=True)

    inp = df.iloc[:, 9:20]
    classes, indices = np.unique(df["Marital_Status"], return_inverse=True)
    assert inp.shape[0] == indices.shape[0]

    inp = inp.to_numpy(dtype=np.float32)
    out = indices.astype(np.int64)

    return (inp, out, classes)


if __name__ == "__main__":
    # data init
    X, Y, classes = data_preproc("marketing_campaign.csv")
    split_idx = int(SPLIT_RATIO * X.shape[0])
    Xtr, Ytr = X[:split_idx], Y[:split_idx]
    Xvl, Yvl = X[split_idx:], Y[split_idx:]

    # mlps sensitive to feature scaling
    mean = np.mean(Xtr, axis=0)
    eps = 1e-8
    std = np.std(Xtr, axis=0) + eps
    Xtr = (Xtr - mean) / std
    Xvl = (Xvl - mean) / std

    # param init kaiming
    W1 = np.random.randn(X.shape[1], 64) * (2 / X.shape[1]) ** 0.5
    b1 = np.random.randn(64) * 0.01
    W2 = np.random.randn(64, 32) * (2 / 64) ** 0.5
    b2 = np.random.randn(32) * 0.01
    W3 = np.random.randn(32, len(classes)) * (1 / 32) ** 0.5
    b3 = np.zeros(len(classes))
    params = [W1, b1, W2, b2, W3, b3]

    n_param = np.sum([i.size for i in params])
    logging.info(f"init complete! total params: {n_param}")

    lossi =[]

    # training SGD
    for step in range(MAX_ITER):
        batch_idx = np.random.randint(0, Xtr.shape[0], (BS,))
        Xb, Yb = Xtr[batch_idx], Ytr[batch_idx]

        # fp
        z1 = np.dot(Xb, W1) + b1
        h1 = np.maximum(0, z1)
        z2 = np.dot(h1, W2) + b2
        h2 = np.maximum(0, z2)
        logits = np.dot(h2, W3) + b3
        # cross entropy combined
        expl = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        sm_prob = expl / expl.sum(1, keepdims=True)
        ll = np.log(np.clip(sm_prob[np.arange(BS), Yb], 1e-12, 1.0))
        loss = -np.mean(ll)
        # cross entropy explicit
        # shifted = logits - np.max(logits, axis=1, keepdims=True)
        # expl = np.exp(shifted)
        # sums = expl.sum(axis=1, keepdims=True)
        # sm_prob = expl / sums
        # correct_probs = sm_prob[np.arange(BS), Yb]
        # ll = np.log(correct_probs)
        # loss = -np.mean(ll)
        if step % 500 == 0:
            logging.info(f"loss at {step}: {loss}")
        lossi.append(loss)
        # bp
        # cross entropy fast (Pi when i!=y , Pi-1 when i=y )
        dlogits = sm_prob.copy()
        dlogits[np.arange(Xb.shape[0]), Yb] -= 1
        dlogits /= Xb.shape[0]
        # cross entropy slow
        # dll = -np.ones_like(ll) / BS
        # dcorrect_probs = dll / correct_probs
        # dsm_prob = np.zeros_like(sm_prob)
        # dsm_prob[np.arange(BS), Yb] = dcorrect_probs
        # dexpl = dsm_prob / sums
        # dsums = np.sum(dsm_prob * (-expl / sums**2), axis=1, keepdims=True)
        # dexpl += np.ones_like(expl) * dsums
        # dshifted = dexpl * expl
        # dlogits = dshifted
        db3 = dlogits.sum(axis=0)
        dW3 = np.dot(np.transpose(h2), dlogits)
        dh2 = np.dot(dlogits, np.transpose(W3))

        dz2 = dh2 * (z2 > 0)
        db2 = dz2.sum(axis=0)
        dW2 = np.dot(np.transpose(h1), dz2)
        dh1 = np.dot(dz2, np.transpose(W2))

        dz1 = dh1 * (z1 > 0)
        db1 = dz1.sum(axis=0)
        dW1 = np.dot(np.transpose(Xb), dz1)

        lr = LR * (1 - step / MAX_ITER)
        # update
        dparams = [dW1, db1, dW2, db2, dW3, db3]
        for i in range(len(params)):
            params[i] -= lr * dparams[i]
        
    # eval
    h1 = np.maximum(0,np.dot(Xvl, W1) + b1)
    h2 = np.maximum(0,np.dot(h1, W2) + b2)
    logits = np.dot(h2, W3) + b3
    pred = np.argmax(logits, axis=1)
    accuracy = np.mean(pred == Yvl)

    # plotting
    logging.info(f"validation accuracy: {accuracy:.4f}")

    #avg for a certain window
    smooth_loss = np.convolve(
        lossi,
        np.ones(WINDOW) / WINDOW,
        mode="valid",
    )
    plt.plot(smooth_loss)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    info = (
        f"accuracy: {accuracy:.4f}\n"
        f"final loss: {lossi[-1]:.4f}\n"
        f"LR: {LR}\n"
        f"batch size: {BS}"
    )
    plt.text(
        0.95,
        0.95,
        info,
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", alpha=0.8),
    )
    plt.savefig("mlp_loss.png")