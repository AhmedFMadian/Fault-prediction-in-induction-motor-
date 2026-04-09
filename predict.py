import os
import glob
import numpy as np
import scipy.io
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks, regularizers
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# CONFIG
# =========================
DATA_DIR     = r"/content/paderborn_data"
WINDOW_SIZE  = 1024
OVERLAP      = 0.5
SAMPLE_RATE  = 64000
EPOCHS       = 100
BATCH_SIZE   = 128
LR           = 0.001
MC_PASSES    = 50
DROPOUT_RATE = 0.4
RANDOM_SEED  = 42
MODEL_PATH   = "rap_msf_best.keras"

LABEL_MAP = {
    "K":  "Healthy",
    "KI": "Inner Race Fault",
    "KA": "Outer Race Fault",
}

# Condition-level split — same bearings, different operating conditions
TRAIN_CONDITIONS = ["N09_M07_F10", "N15_M07_F10"]
VAL_CONDITIONS   = ["N15_M01_F10"]
TEST_CONDITIONS  = ["N15_M07_F04"]

ALL_BEARINGS = [
    "K001","K002","K003","K004","K005","K006",
    "KA01","KA03","KA04","KA05","KA06","KA07",
    "KA08","KA09","KA15","KA16","KA22","KA30",
    "KI01","KI03","KI04","KI05","KI07","KI08",
    "KI14","KI16","KI17","KI18","KI21",
]

# ── Targeted frequency bands ─────────────────────────────────
# From signal analysis:
#   Inner race fault  → 375–625 Hz   (low band)
#   Outer race fault  → 5688–6062 Hz (high band)
#   Both bands together give the CNN exactly the discriminative regions
FREQ_RES   = SAMPLE_RATE / WINDOW_SIZE          # 62.5 Hz per bin
LOW_START  = int(375  / FREQ_RES)               # bin ~6
LOW_END    = int(650  / FREQ_RES)               # bin ~10  (+margin)
HIGH_START = int(5500 / FREQ_RES)               # bin ~88
HIGH_END   = int(6500 / FREQ_RES)               # bin ~104 (+margin)

# Also include a mid-band for general health monitoring
MID_START  = int(1000 / FREQ_RES)               # bin ~16
MID_END    = int(3000 / FREQ_RES)               # bin ~48

print(f"Low band  : {LOW_START*FREQ_RES:.0f}–{LOW_END*FREQ_RES:.0f} Hz  (bins {LOW_START}–{LOW_END})")
print(f"Mid band  : {MID_START*FREQ_RES:.0f}–{MID_END*FREQ_RES:.0f} Hz (bins {MID_START}–{MID_END})")
print(f"High band : {HIGH_START*FREQ_RES:.0f}–{HIGH_END*FREQ_RES:.0f} Hz (bins {HIGH_START}–{HIGH_END})")

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =========================
# SIGNAL EXTRACTION
# =========================
def extract_signals_from_mat(filepath):
    try:
        mat = scipy.io.loadmat(filepath, simplify_cells=True)
        main_key = [k for k in mat.keys() if not k.startswith("__")][0]
        data = mat[main_key]
        if "Y" not in data:
            return None, None
        Y = data["Y"]
        if not isinstance(Y, list):
            return None, None
        vib, cur = None, None
        for sensor in Y:
            name = str(sensor.get("Name", "")).strip()
            if name == "vibration_1":
                vib = np.array(sensor["Data"]).ravel().astype(np.float32)
            elif name == "phase_current_1":
                cur = np.array(sensor["Data"]).ravel().astype(np.float32)
        if vib is None or cur is None:
            return None, None
        return vib, cur
    except Exception as e:
        print(f"  [ERROR] {filepath}: {e}")
        return None, None


def segment_signal(signal, window, overlap):
    if len(signal) < window: return None
    step = int(window * (1 - overlap))
    segments = [signal[i:i+window]
                for i in range(0, len(signal)-window+1, step)]
    return np.stack(segments) if segments else None


def get_bearing_id(filepath):
    return os.path.basename(os.path.dirname(filepath)).upper()


def get_condition(filepath):
    parts = os.path.basename(filepath).split("_")
    return "_".join(parts[:3])


def get_label_from_bearing(bearing_id):
    if bearing_id.startswith("KI"): return "KI"
    if bearing_id.startswith("KA"): return "KA"
    if bearing_id.startswith("KB"): return None
    if bearing_id.startswith("K"):  return "K"
    return None


def compute_targeted_psd(windows):
    """
    Extract PSD from 3 targeted frequency bands only:
      - Low  (375–650 Hz)  : inner race fault harmonics
      - Mid  (1–3 kHz)     : general structural health
      - High (5.5–6.5 kHz) : outer race fault harmonics
    Concatenate bands → compact discriminative feature vector
    """
    hann     = np.hanning(windows.shape[1])
    fft_vals = np.fft.rfft(windows * hann, axis=1)
    psd      = (np.abs(fft_vals) ** 2) / windows.shape[1]
    psd      = np.log1p(psd)

    low  = psd[:, LOW_START:LOW_END]
    mid  = psd[:, MID_START:MID_END]
    high = psd[:, HIGH_START:HIGH_END]

    return np.concatenate([low, mid, high], axis=1).astype(np.float32)


# =========================
# DATA LOADING
# =========================
def load_split(data_dir, condition_list, split_name):
    mat_files = sorted(glob.glob(
        os.path.join(data_dir, "**", "*.mat"), recursive=True))

    all_vib, all_cur, all_labels = [], [], []

    for fpath in mat_files:
        bearing_id = get_bearing_id(fpath)
        condition  = get_condition(fpath)

        if bearing_id not in ALL_BEARINGS: continue
        if condition not in condition_list: continue

        label = get_label_from_bearing(bearing_id)
        if label is None: continue

        vib, cur = extract_signals_from_mat(fpath)
        if vib is None: continue

        vib_w = segment_signal(vib, WINDOW_SIZE, OVERLAP)
        cur_w = segment_signal(cur, WINDOW_SIZE, OVERLAP)
        if vib_w is None or cur_w is None: continue

        n = min(len(vib_w), len(cur_w))
        all_vib.append(compute_targeted_psd(vib_w[:n]))
        all_cur.append(compute_targeted_psd(cur_w[:n]))
        all_labels.extend([label] * n)

    if not all_vib:
        raise ValueError(f"No data for {split_name}!")

    X_vib = np.concatenate(all_vib, axis=0)
    X_cur  = np.concatenate(all_cur, axis=0)
    y      = np.array(all_labels)

    unique, counts = np.unique(y, return_counts=True)
    print(f"  {split_name}: {len(y):>7} windows | " +
          " | ".join(f"{LABEL_MAP[u]}: {c}" for u, c in zip(unique, counts)))
    return X_vib, X_cur, y


# =========================
# PREPROCESSING
# =========================
def preprocess(X_vib_tr, X_cur_tr,
               X_vib_val, X_cur_val,
               X_vib_te, X_cur_te,
               y_tr, y_val, y_te):
    sv = MinMaxScaler()
    sc = MinMaxScaler()

    X_vib_tr  = sv.fit_transform(X_vib_tr)[..., np.newaxis]
    X_vib_val = sv.transform(X_vib_val)[..., np.newaxis]
    X_vib_te  = sv.transform(X_vib_te)[..., np.newaxis]

    X_cur_tr  = sc.fit_transform(X_cur_tr)[..., np.newaxis]
    X_cur_val = sc.transform(X_cur_val)[..., np.newaxis]
    X_cur_te  = sc.transform(X_cur_te)[..., np.newaxis]

    le    = LabelEncoder()
    y_tr  = le.fit_transform(y_tr)
    y_val = le.transform(y_val)
    y_te  = le.transform(y_te)

    return (X_vib_tr, X_cur_tr,
            X_vib_val, X_cur_val,
            X_vib_te, X_cur_te,
            y_tr, y_val, y_te, le)


# =========================
# MODEL
# =========================
def cnn_branch(input_shape, prefix):
    l2  = regularizers.l2(1e-4)
    inp = layers.Input(shape=input_shape, name=f"{prefix}_input")

    x = layers.Conv1D(32, 3, padding="same", kernel_regularizer=l2)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(64, 3, padding="same", kernel_regularizer=l2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(128, 3, padding="same", kernel_regularizer=l2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling1D(name=f"{prefix}_gap")(x)

    return inp, x


def build_model(input_dim, n_classes):
    shape = (input_dim, 1)
    v_in, v_feat = cnn_branch(shape, "vib")
    c_in, c_feat = cnn_branch(shape, "cur")

    fused = layers.Concatenate(name="fusion")([v_feat, c_feat])

    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(fused)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_RATE, name="bayesian_drop_1")(x)

    x = layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_RATE, name="bayesian_drop_2")(x)

    out = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = Model([v_in, c_in], out, name="RAP_MSF_v12")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# =========================
# MC DROPOUT
# =========================
def mc_predict(model, X_vib, X_cur, T=MC_PASSES):
    preds = np.stack(
        [model([X_vib, X_cur], training=True).numpy() for _ in range(T)],
        axis=0
    )
    mu      = preds.mean(axis=0)
    eps     = 1e-8
    entropy = -np.sum(mu * np.log(mu + eps), axis=1)
    return mu, entropy


# =========================
# VISUALISATION
# =========================
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"],     label="Train")
    axes[0].plot(history.history["val_loss"], label="Val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(history.history["accuracy"],     label="Train")
    axes[1].plot(history.history["val_accuracy"], label="Val")
    axes[1].set_title("Accuracy"); axes[1].legend()
    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("RAP-MSF v12 — Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()

def plot_entropy_violin(entropy, y_true, y_pred):
    correct   = entropy[y_true == y_pred]
    incorrect = entropy[y_true != y_pred]
    fig, ax   = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot([correct, incorrect], showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(["#4CAF50", "#F44336"][i])
        pc.set_alpha(0.75)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"Correct (n={len(correct)})",
                        f"Incorrect (n={len(incorrect)})"])
    ax.set_ylabel("Predictive Entropy (H)")
    ax.set_title("Uncertainty: Correct vs Incorrect Predictions")
    plt.tight_layout()
    plt.savefig("entropy_violin.png", dpi=150)
    plt.show()


# =========================
# MAIN
# =========================
def main():
    print("[1/4] Loading Data (condition-level split, targeted PSD)...")
    print(f"  Train: {TRAIN_CONDITIONS}")
    print(f"  Val  : {VAL_CONDITIONS}")
    print(f"  Test : {TEST_CONDITIONS}\n")

    X_vib_tr,  X_cur_tr,  y_tr  = load_split(DATA_DIR, TRAIN_CONDITIONS, "Train")
    X_vib_val, X_cur_val, y_val = load_split(DATA_DIR, VAL_CONDITIONS,   "Val  ")
    X_vib_te,  X_cur_te,  y_te  = load_split(DATA_DIR, TEST_CONDITIONS,  "Test ")

    print("\n[2/4] Preprocessing...")
    (X_vib_tr, X_cur_tr,
     X_vib_val, X_cur_val,
     X_vib_te, X_cur_te,
     y_tr, y_val, y_te, le) = preprocess(
        X_vib_tr, X_cur_tr,
        X_vib_val, X_cur_val,
        X_vib_te, X_cur_te,
        y_tr, y_val, y_te
    )
    class_names = [LABEL_MAP[c] for c in le.classes_]
    print(f"  Feature vector size per sensor: {X_vib_tr.shape[1]} bins")
    print(f"  Input shape: {X_vib_tr.shape}")

    weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_tr), y=y_tr)
    cw_dict = dict(enumerate(weights))
    print(f"  Class weights: {cw_dict}")

    print("\n[3/4] Building Model...")
    model = build_model(X_vib_tr.shape[1], len(le.classes_))
    model.summary()

    cb_list = [
        callbacks.EarlyStopping(monitor='val_accuracy', patience=15,
                                restore_best_weights=True, verbose=1,
                                mode='max'),
        callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_accuracy',
                                  save_best_only=True, verbose=1,
                                  mode='max'),
        callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5,
                                    patience=7, min_lr=1e-6, verbose=1,
                                    mode='max'),
    ]

    print("\n[4/4] Training...")
    history = model.fit(
        [X_vib_tr, X_cur_tr], y_tr,
        validation_data=([X_vib_val, X_cur_val], y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=cw_dict,
        callbacks=cb_list,
        verbose=1
    )

    

    plot_history(history)

    print(f"\n[EVAL] MC-Dropout inference (T={MC_PASSES})...")
    mu, entropy = mc_predict(model, X_vib_te, X_cur_te)
    y_pred = np.argmax(mu, axis=1)

    print("\n── Classification Report ──────────────────────────")
    print(classification_report(y_te, y_pred, target_names=class_names))
    print(f"Mean Entropy (all)       : {entropy.mean():.4f}")
    print(f"Mean Entropy (correct)   : {entropy[y_te == y_pred].mean():.4f}")
    print(f"Mean Entropy (incorrect) : {entropy[y_te != y_pred].mean():.4f}")

    plot_confusion_matrix(y_te, y_pred, class_names)
    plot_entropy_violin(entropy, y_te, y_pred)


if __name__ == "__main__":
    
    main()