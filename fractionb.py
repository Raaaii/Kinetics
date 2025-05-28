import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lmfit import Model
from sklearn.utils import resample
from scipy.stats import norm
from matplotlib.backends.backend_pdf import PdfPages
import os

# === Setup output folder ===
output_dir = "fit_data"
os.makedirs(output_dir, exist_ok=True)


# === Load and preprocess data ===
def load_data():
    data = {
        "Conc ATP": [0.5] * 9,
        "Conc HSA": [0.0, 0.1, 0.2, 0.4, 0.8, 1.0, 1.5, 2.0, 2.4],
        "Area DMSO": [
            2511.2,
            2511.2,
            2101.159,
            2388.41,
            2750.91,
            2446.67,
            2397.34,
            3176.93,
            2376.6,
        ],
        "Area ATR-HSA": [
            1624.44,
            1389.53,
            1047.34,
            1150.82,
            1004.09,
            629.466,
            504.635,
            367.919,
            220.813,
        ],
    }
    df = pd.DataFrame(data)
    df["Normalized Absorbance"] = df["Area ATR-HSA"] / df["Area DMSO"]
    df["Fraction Bound"] = 1 - df["Normalized Absorbance"]
    df["[HSA] (µM)"] = df["Conc HSA"] * 1000
    return df


# === Define models ===
def binding_model(x, Kd, Bmax):
    return baseline_fixed + (Bmax * x) / (Kd + x)


def binding_model_fixed_bmax(x, Kd):
    return baseline_fixed + (BMAX_FIXED * x) / (Kd + x)


# === Bootstrap resampling ===
def bootstrap_fit(model, x_data, y_data, n=500):
    boot_kd, boot_bmax = [], []
    for _ in range(n):
        x_boot, y_boot = resample(x_data, y_data)
        try:
            res = model.fit(y_boot, model.make_params(Kd=1000, Bmax=1), x=x_boot)
            if 100 < res.best_values["Kd"] < 5000:
                boot_kd.append(res.best_values["Kd"])
                boot_bmax.append(res.best_values["Bmax"])
        except:
            continue
    return np.array(boot_kd), np.array(boot_bmax)


# === Main analysis ===
df = load_data()
x_data = df["[HSA] (µM)"].values
y_data = df["Fraction Bound"].values
baseline_fixed = y_data[0]
BMAX_FIXED = 1.0
x_fit = np.linspace(0, max(x_data), 300)

# Fit full model
model = Model(binding_model)
params = model.make_params(Kd=1000, Bmax=1)
result = model.fit(y_data, params, x=x_data)
y_fit = result.eval(x=x_fit)

# Fit Kd-only model
model_kd_only = Model(binding_model_fixed_bmax)
params_kd_only = model_kd_only.make_params(Kd=1000)
result_kd_only = model_kd_only.fit(y_data, params_kd_only, x=x_data)
y_fit_kd_only = result_kd_only.eval(x=x_fit)

# Bootstrap analysis
boot_kd, boot_bmax = bootstrap_fit(model, x_data, y_data)

# Confidence intervals
ci_kd = np.percentile(boot_kd, [2.5, 97.5])
ci_bmax = np.percentile(boot_bmax, [2.5, 97.5])
mu_kd, std_kd = norm.fit(boot_kd)
mu_bmax, std_bmax = norm.fit(boot_bmax)
ci_kd_param = [mu_kd - 1.96 * std_kd, mu_kd + 1.96 * std_kd]
ci_bmax_param = [mu_bmax - 1.96 * std_bmax, mu_bmax + 1.96 * std_bmax]



# === Plot: Fit comparison ===
plt.figure(figsize=(8, 5))
plt.scatter(x_data, y_data, color="navy", label="Experimental Data")

# Plot full model (free Bmax)
plt.plot(
    x_fit,
    y_fit,
    "r--",
    label=f"Fit (Kd+Bmax): Kd={result.best_values['Kd'] / 1000:.2f} mM, "
          f"Bmax={result.best_values['Bmax']:.2f}"
)

# Plot Kd-only model (fixed Bmax = 1)
plt.plot(
    x_fit,
    y_fit_kd_only,
    "g--",
    label=f"Fit (Kd only, Bmax=1): Kd={result_kd_only.best_values['Kd'] / 1000:.2f} mM"
)

# Optional: theoretical max fraction (dashed line)
plt.axhline(1.0, color="gray", linestyle=":", alpha=0.5, label="Bmax = 1")

plt.xlabel("[HSA] (µM)")
plt.ylabel("Fraction of ATP Bound")
plt.title("Model Fit Comparison")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"{output_dir}/fit_comparison.png")
plt.show()



# === Plot: Fit comparison ===
plt.figure(figsize=(8, 5))
plt.scatter(x_data, y_data, color="navy", label="Experimental Data")
# plt.plot(x_fit, y_fit, "r--", label=f"Fit (Kd+Bmax): Kd={result.best_values['Kd']:.1f}")
plt.plot(
    x_fit,
    y_fit_kd_only,
    "g--",
    label=f"Fit Kd: Kd={result_kd_only.best_values['Kd'] / 1000:.2f} mM",
)

# plt.axhline(
#    baseline_fixed + BMAX_FIXED, color="gray", linestyle=":", label="Fixed Max Fraction")
plt.xlabel("[HSA] (µM)")
plt.ylabel("Fraction of ATP Bound")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

plt.savefig(f"{output_dir}/fit_comparison.png")
plt.close()

# === Plot: Residuals ===
residuals = y_data - result.best_fit
plt.figure(figsize=(6, 4))
plt.stem(x_data, residuals, basefmt=" ")
plt.title("Residuals of the Fit")
plt.xlabel("[HSA] (µM)")
plt.ylabel("Residual")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

plt.savefig(f"{output_dir}/residuals.png")
plt.close()

# === Plot: Bootstrap distributions ===
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.histplot(boot_kd, kde=True, color="forestgreen")
plt.axvline(ci_kd[0], color="gray", linestyle="--")
plt.axvline(ci_kd[1], color="gray", linestyle="--")
plt.title("Bootstrap Kd Distribution")
plt.xlabel("Kd (µM)")
plt.subplot(1, 2, 2)
sns.histplot(boot_bmax, kde=True, color="darkorange")
plt.axvline(ci_bmax[0], color="gray", linestyle="--")
plt.axvline(ci_bmax[1], color="gray", linestyle="--")
plt.title("Bootstrap Bmax Distribution")
plt.xlabel("Bmax")

plt.tight_layout()
plt.show()

plt.savefig(f"{output_dir}/bootstrap_distributions.png")
plt.close()

# === Save fitting report to PDF ===
with PdfPages(f"{output_dir}/fitting_statistics.pdf") as pdf:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    text = f"""
=== Fit Report: Full Model ===
{result.fit_report()}

=== Fit Report: Kd-only Model ===
{result_kd_only.fit_report()}

=== 95% Confidence Intervals (Bootstrap) ===
Kd: {ci_kd[0]:.1f} – {ci_kd[1]:.1f} µM


=== 95% Confidence Intervals (Normal Fit) ===
Kd: {ci_kd_param[0]:.1f} – {ci_kd_param[1]:.1f} µM

"""
    ax.text(0.01, 0.99, text, ha="left", va="top", family="monospace", fontsize=8)
    plt.tight_layout()
    plt.show()

    pdf.savefig(fig)
    plt.close()
