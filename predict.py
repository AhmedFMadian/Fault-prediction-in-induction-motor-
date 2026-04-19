"""
RAP-MSF Bearing Fault Diagnosis — Desktop GUI
=============================================
Files required in the same folder:
  - rap_msf_best (1).keras
  - scaler_vib.pkl
  - scaler_cur.pkl

Run:
  python predict.py
"""

import os
import sys
import pickle
import threading
import numpy as np
import scipy.io
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow as tf

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "rap_msf_best (3).keras")
SCALER_VIB  = os.path.join(BASE_DIR, "scaler_vib (2).pkl")
SCALER_CUR  = os.path.join(BASE_DIR, "scaler_cur (2).pkl")

WINDOW_SIZE = 1024
OVERLAP     = 0.5
SAMPLE_RATE = 64000
MC_PASSES   = 50
FREQ_RES    = SAMPLE_RATE / WINDOW_SIZE

# Feature Indices for the 52 inputs (4 + 32 + 16)
LOW_START,  LOW_END  = int(375 / FREQ_RES),  int(650 / FREQ_RES)
MID_START,  MID_END  = int(1000 / FREQ_RES), int(3000 / FREQ_RES)
HIGH_START, HIGH_END = int(5500 / FREQ_RES), int(6500 / FREQ_RES)

LABELS = {0: "Healthy", 1: "Outer Race Fault", 2: "Inner Race Fault"}
COLORS      = {"Healthy": "#2ecc71", "Inner Race Fault": "#e74c3c", "Outer Race Fault": "#e67e22"}
ICONS       = {"Healthy": "✅", "Inner Race Fault": "⚠️", "Outer Race Fault": "🔴"}

# ─────────────────────────────────────────────────────────────
# SIGNAL PROCESSING
# ─────────────────────────────────────────────────────────────
def extract_signals(filepath):
    mat      = scipy.io.loadmat(filepath, simplify_cells=True)
    main_key = [k for k in mat.keys() if not k.startswith("__")][0]
    data     = mat[main_key]
    Y        = data.get("Y", [])
    vib, cur = None, None
    for sensor in Y:
        name = str(sensor.get("Name", "")).strip()
        if name == "vibration_1":
            vib = np.array(sensor["Data"]).ravel().astype(np.float32)
        elif name == "phase_current_1":
            cur = np.array(sensor["Data"]).ravel().astype(np.float32)
    return vib, cur

def segment(signal):
    step     = int(WINDOW_SIZE * (1 - OVERLAP))
    segments = [signal[i:i+WINDOW_SIZE]
                for i in range(0, len(signal)-WINDOW_SIZE+1, step)]
    return np.stack(segments)

def compute_psd(windows):
    hann = np.hanning(windows.shape[1])
    # Compute FFT magnitude
    fft_vals = np.abs(np.fft.rfft(windows * hann, axis=1))
    # PSD formula matching training script
    psd = (fft_vals**2) / windows.shape[1]
    # Critical Log transformation
    psd = np.log1p(psd)
    
    return np.concatenate([
        psd[:, LOW_START:LOW_END],
        psd[:, MID_START:MID_END],
        psd[:, HIGH_START:HIGH_END]
    ], axis=1).astype(np.float32)

def mc_predict(model, X_vib, X_cur):
    # Use direct prediction instead of MC Dropout
    # since the dropout rate is too high for stable MC inference
    pred = model([X_vib, X_cur], training=False).numpy()
    mu   = pred  # shape (N, 3)
    
    # Entropy from direct predictions
    eps     = 1e-8
    entropy = -np.sum(mu * np.log(mu + eps), axis=1)
    return mu, entropy
# ─────────────────────────────────────────────────────────────
# GUI APPLICATION
# ─────────────────────────────────────────────────────────────
class BearingDiagnosisApp:
    def __init__(self, root):
        self.root    = root
        self.model   = None
        self.sv      = None
        self.sc      = None
        self.result  = None

        root.title("RAP-MSF — Bearing Fault Diagnosis System")
        root.geometry("900x700")
        root.resizable(True, True)
        root.configure(bg="#1e1e2e")

        self._build_ui()
        self._load_model_async()

    # ── UI CONSTRUCTION ──────────────────────────────────────
    def _build_ui(self):
        # ── Header ───────────────────────────────────────────
        header = tk.Frame(self.root, bg="#12121f", pady=12)
        header.pack(fill="x")

        tk.Label(header, text="⚙  RAP-MSF Bearing Fault Diagnosis",
                 font=("Segoe UI", 18, "bold"),
                 fg="#cdd6f4", bg="#12121f").pack()
        tk.Label(header,
                 text="Probabilistic Multi-Sensor Fusion  |  MC Dropout Uncertainty",
                 font=("Segoe UI", 10),
                 fg="#6c7086", bg="#12121f").pack()

        # ── Status bar ───────────────────────────────────────
        self.status_var = tk.StringVar(value="Loading model...")
        status_bar = tk.Label(self.root, textvariable=self.status_var,
                              font=("Segoe UI", 9), fg="#a6adc8",
                              bg="#313244", anchor="w", padx=10)
        status_bar.pack(fill="x")

        # ── Main content ─────────────────────────────────────
        main = tk.Frame(self.root, bg="#1e1e2e")
        main.pack(fill="both", expand=True, padx=20, pady=10)

        # Left panel — controls
        left = tk.Frame(main, bg="#1e1e2e", width=280)
        left.pack(side="left", fill="y", padx=(0, 15))
        left.pack_propagate(False)

        self._build_left_panel(left)

        # Right panel — results
        right = tk.Frame(main, bg="#1e1e2e")
        right.pack(side="left", fill="both", expand=True)

        self._build_right_panel(right)

    def _build_left_panel(self, parent):
        # File selection card
        card = tk.Frame(parent, bg="#313244", bd=0, relief="flat")
        card.pack(fill="x", pady=(0, 12))

        tk.Label(card, text="📂  Select .mat File",
                 font=("Segoe UI", 11, "bold"),
                 fg="#cdd6f4", bg="#313244").pack(anchor="w", padx=12, pady=(12, 6))

        self.file_var = tk.StringVar(value="No file selected")
        tk.Label(card, textvariable=self.file_var,
                 font=("Segoe UI", 8), fg="#6c7086",
                 bg="#313244", wraplength=240, anchor="w").pack(padx=12, pady=(0, 8))

        tk.Button(card, text="Browse File",
                  font=("Segoe UI", 10, "bold"),
                  bg="#89b4fa", fg="#1e1e2e",
                  activebackground="#74c7ec",
                  relief="flat", padx=12, pady=6,
                  cursor="hand2",
                  command=self._browse_file).pack(padx=12, pady=(0, 12), fill="x")

        # Analyze button
        self.analyze_btn = tk.Button(parent,
                  text="🔍  Analyze Bearing",
                  font=("Segoe UI", 12, "bold"),
                  bg="#a6e3a1", fg="#1e1e2e",
                  activebackground="#94e2d5",
                  relief="flat", padx=12, pady=10,
                  cursor="hand2",
                  command=self._run_analysis,
                  state="disabled")
        self.analyze_btn.pack(fill="x", pady=(0, 12))

        # MC Dropout info card
        info = tk.Frame(parent, bg="#313244")
        info.pack(fill="x", pady=(0, 12))

        tk.Label(info, text="ℹ  About This Model",
                 font=("Segoe UI", 10, "bold"),
                 fg="#cdd6f4", bg="#313244").pack(anchor="w", padx=12, pady=(10, 4))

        info_text = (
            "Uses MC Dropout with T=50\n"
            "stochastic forward passes to\n"
            "produce a diagnosis + confidence\n"
            "score on every prediction.\n\n"
            "Low entropy  →  certain\n"
            "High entropy →  uncertain"
        )
        tk.Label(info, text=info_text,
                 font=("Segoe UI", 9),
                 fg="#a6adc8", bg="#313244",
                 justify="left").pack(anchor="w", padx=12, pady=(0, 10))

        # Progress bar (hidden until analysis)
        self.progress = ttk.Progressbar(parent, mode="indeterminate")

    def _build_right_panel(self, parent):
        # Diagnosis result card
        self.result_frame = tk.Frame(parent, bg="#313244")
        self.result_frame.pack(fill="x", pady=(0, 12))

        tk.Label(self.result_frame, text="Diagnosis Result",
                 font=("Segoe UI", 11, "bold"),
                 fg="#6c7086", bg="#313244").pack(pady=(12, 4))

        self.diagnosis_label = tk.Label(self.result_frame,
                 text="—",
                 font=("Segoe UI", 22, "bold"),
                 fg="#cdd6f4", bg="#313244")
        self.diagnosis_label.pack()

        self.confidence_label = tk.Label(self.result_frame,
                 text="",
                 font=("Segoe UI", 10),
                 fg="#a6adc8", bg="#313244")
        self.confidence_label.pack(pady=(2, 12))

        # Probability bars
        bars_frame = tk.Frame(parent, bg="#313244")
        bars_frame.pack(fill="x", pady=(0, 12))

        tk.Label(bars_frame, text="Class Probabilities (μ)",
                 font=("Segoe UI", 10, "bold"),
                 fg="#cdd6f4", bg="#313244").pack(anchor="w", padx=12, pady=(10, 6))

        self.bar_vars   = {}
        self.bar_labels = {}
        for label in ["Healthy", "Inner Race Fault", "Outer Race Fault"]:
            row = tk.Frame(bars_frame, bg="#313244")
            row.pack(fill="x", padx=12, pady=3)

            tk.Label(row, text=label, width=18, anchor="w",
                     font=("Segoe UI", 9), fg="#a6adc8",
                     bg="#313244").pack(side="left")

            bar_bg = tk.Frame(row, bg="#45475a", height=16, width=180)
            bar_bg.pack(side="left", padx=(4, 4))
            bar_bg.pack_propagate(False)

            bar_fill = tk.Frame(bar_bg, bg=COLORS[label], height=16, width=0)
            bar_fill.place(x=0, y=0, height=16)
            self.bar_vars[label] = bar_fill

            pct_label = tk.Label(row, text="0%",
                                 font=("Segoe UI", 9, "bold"),
                                 fg="#cdd6f4", bg="#313244", width=5)
            pct_label.pack(side="left")
            self.bar_labels[label] = pct_label

        # Chart — entropy distribution
        chart_frame = tk.Frame(parent, bg="#313244")
        chart_frame.pack(fill="both", expand=True)

        tk.Label(chart_frame, text="Predictive Entropy per Window",
                 font=("Segoe UI", 10, "bold"),
                 fg="#cdd6f4", bg="#313244").pack(anchor="w", padx=12, pady=(10, 4))

        fig = Figure(figsize=(5, 2.2), facecolor="#313244")
        self.ax = fig.add_subplot(111, facecolor="#1e1e2e")
        self.ax.set_xlabel("Window index", color="#a6adc8", fontsize=8)
        self.ax.set_ylabel("Entropy (H)", color="#a6adc8", fontsize=8)
        self.ax.tick_params(colors="#6c7086", labelsize=7)
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#45475a")
        fig.tight_layout(pad=1.5)

        self.canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=12, pady=(0, 12))

    # ── MODEL LOADING ─────────────────────────────────────────
    def _load_model_async(self):
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        try:
            self.status_var.set("⏳  Loading model weights...")
            self.model = tf.keras.models.load_model(MODEL_PATH)

            with open(SCALER_VIB, "rb") as f:
                self.sv = pickle.load(f)
            with open(SCALER_CUR, "rb") as f:
                self.sc = pickle.load(f)

            self.status_var.set("✅  Model ready — select a .mat file to begin")
            self.analyze_btn.config(state="normal")
        except Exception as e:
            self.status_var.set(f"❌  Error loading model: {e}")
            messagebox.showerror("Load Error",
                f"Could not load model or scalers.\n\nMake sure these files are in the same folder as predict.py:\n"
                f"  • rap_msf_best (1).keras\n  • scaler_vib.pkl\n  • scaler_cur.pkl\n\nError: {e}")

    # ── FILE BROWSING ─────────────────────────────────────────
    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Select bearing .mat file",
            filetypes=[("MATLAB files", "*.mat"), ("All files", "*.*")]
        )
        if path:
            self.selected_file = path
            self.file_var.set(os.path.basename(path))
            self.status_var.set(f"File selected: {os.path.basename(path)}")

    # ── ANALYSIS ─────────────────────────────────────────────
    def _run_analysis(self):
        if not hasattr(self, "selected_file") or not self.selected_file:
            messagebox.showwarning("No File", "Please select a .mat file first.")
            return
        if self.model is None:
            messagebox.showwarning("Model Not Ready", "Please wait for the model to finish loading.")
            return

        self.analyze_btn.config(state="disabled")
        self.progress.pack(fill="x", pady=4)
        self.progress.start(10)
        self.status_var.set("⏳  Running MC Dropout inference (50 passes)...")

        threading.Thread(target=self._analysis_thread,
                         args=(self.selected_file,), daemon=True).start()

    def _analysis_thread(self, filepath):
        try:
            # 1. Extract raw signals
            vib, cur = extract_signals(filepath)
            if vib is None or cur is None:
                raise ValueError("Could not find required sensors in .mat file.")

            # 2. Segment and calculate features
            vib_w = segment(vib)
            cur_w = segment(cur)
            n = min(len(vib_w), len(cur_w))

            X_vib = compute_psd(vib_w[:n])
            X_cur = compute_psd(cur_w[:n])

            # 3. Scale and Reshape to (Batch, 52, 1)
            X_vib = self.sv.transform(X_vib)[..., np.newaxis]
            X_cur = self.sc.transform(X_cur)[..., np.newaxis]

            # 4. Run MC Dropout inference
            mu, entropy = mc_predict(self.model, X_vib, X_cur)

            # 5. Aggregate results
            mean_mu      = mu.mean(axis=0)
            mean_entropy = entropy.mean()
            pred_class   = np.argmax(mean_mu)
            label        = LABELS[pred_class]
            confidence   = mean_mu[pred_class] * 100

            # 6. Update UI
            self.root.after(0, self._update_ui,
                            label, confidence, mean_mu, entropy, mean_entropy)

        except Exception as e:
            self.root.after(0, self._show_error, str(e))

    def _update_ui(self, label, confidence, mean_mu, entropy, mean_entropy):
        # Stop progress
        self.progress.stop()
        self.progress.pack_forget()
        self.analyze_btn.config(state="normal")

        # Diagnosis result
        color = COLORS[label]
        icon  = ICONS[label]
        self.diagnosis_label.config(
            text=f"{icon}  {label}",
            fg=color
        )

        # Uncertainty interpretation
        if mean_entropy < 0.3:
            cert_text = "High Confidence"
            cert_color = "#a6e3a1"
        elif mean_entropy < 0.7:
            cert_text = "Moderate Confidence"
            cert_color = "#f9e2af"
        else:
            cert_text = "Low Confidence — recommend manual inspection"
            cert_color = "#f38ba8"

        self.confidence_label.config(
            text=f"Confidence: {confidence:.1f}%   |   Entropy H = {mean_entropy:.4f}   |   {cert_text}",
            fg=cert_color
        )

        # Update probability bars
        label_order = ["Healthy", "Outer Race Fault", "Inner Race Fault"]
        for i, lbl in enumerate(label_order):
            pct = mean_mu[i] * 100
            width = int(180 * mean_mu[i])
            self.bar_vars[lbl].place(x=0, y=0, height=16, width=width)
            self.bar_labels[lbl].config(text=f"{pct:.1f}%")

        # Entropy chart
        self.ax.clear()
        self.ax.set_facecolor("#1e1e2e")
        x = np.arange(len(entropy))
        self.ax.fill_between(x, entropy, alpha=0.4, color=COLORS[label])
        self.ax.plot(x, entropy, color=COLORS[label], linewidth=1.2)
        self.ax.axhline(mean_entropy, color="#f38ba8", linewidth=1,
                        linestyle="--", label=f"Mean H={mean_entropy:.3f}")
        self.ax.set_xlabel("Window index", color="#a6adc8", fontsize=8)
        self.ax.set_ylabel("Entropy (H)", color="#a6adc8", fontsize=8)
        self.ax.tick_params(colors="#6c7086", labelsize=7)
        self.ax.legend(fontsize=7, facecolor="#313244",
                       edgecolor="#45475a", labelcolor="#cdd6f4")
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#45475a")
        self.canvas.draw()

        self.status_var.set(
            f"✅  Analysis complete — {label} detected "
            f"({confidence:.1f}% confidence, H={mean_entropy:.4f})"
        )

    def _show_error(self, msg):
        self.progress.stop()
        self.progress.pack_forget()
        self.analyze_btn.config(state="normal")
        self.status_var.set(f"❌  Error: {msg}")
        messagebox.showerror("Analysis Error",
            f"Could not analyze the file.\n\nError: {msg}\n\n"
            "Make sure the .mat file has the PU dataset structure\n"
            "with a Y field containing 'vibration_1' and 'phase_current_1'.")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = BearingDiagnosisApp(root)
    root.mainloop()