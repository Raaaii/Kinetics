import pandas as pd
import matplotlib.pyplot as plt
# data_loader.py or wherever your function is
from kinetics.config import COLUMN_ICG, COLUMN_HSA, COLUMN_DABS, COLUMN_DABS_ERR
from kinetics.config import STANDARD_LIGAND, STANDARD_PROTEIN, STANDARD_ABS


# def load_data(filepath):
#     df = pd.read_excel(filepath)
#     df = df.rename(columns={"B": "[HSA]", "A": "[ICG]", "m": "Delta Abs"})
#     df.dropna(subset=["[HSA]", "[ICG]", "Delta Abs"], inplace=True)
#     return df[["[HSA]", "[ICG]", "Delta Abs"]]

# kinetics/utils.py (updated with flexible print and save functionality)

import pandas as pd
import os


def print_fit_params(params, model="2-Species"):
    print(f"\n=== Best-Fit Parameters ({model}) ===")
    for name, par in params.items():
        print(f"{name:7} = {par.value:.6g}")



def summarize_results(
    model_name,
    Bvals,
    Abs_meas,
    Abs_fit,
    A_free_arr,
    AB_arr,
    A2B_arr,
    FB_arr,
    save_path,
    A3B_arr=None
):
    data = {
        "HSA (M)": Bvals,
        "Delta Abs (meas)": Abs_meas,
        "Delta Abs (fit)": Abs_fit,
        "[ICG_free]": A_free_arr,
        "[HSA·ICG]": AB_arr,
        "[HSA·ICG2]": A2B_arr,
        "Fraction Bound": FB_arr,
    }

    if A3B_arr is not None:
        data["[HSA·ICG3]"] = A3B_arr

    df = pd.DataFrame(data)
    print(f"\n=== Final Results ({model_name}) ===")
    print(df.head(10))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Results saved to '{save_path}'.")

    return df



def load_data_and_build_list(filepath, sheet):
    df = pd.read_excel(filepath, sheet_name=sheet)

    df[STANDARD_PROTEIN] = pd.to_numeric(df[COLUMN_HSA], errors="coerce")
    df[STANDARD_LIGAND] = pd.to_numeric(df[COLUMN_ICG], errors="coerce")
    df[STANDARD_ABS] = pd.to_numeric(df[COLUMN_DABS], errors="coerce")
    df.dropna(subset=[STANDARD_PROTEIN, STANDARD_LIGAND, STANDARD_ABS], inplace=True)

    if COLUMN_DABS_ERR not in df.columns:
        df[COLUMN_DABS_ERR] = 0.0

    data_list = [(row[STANDARD_LIGAND], row[STANDARD_PROTEIN], row[STANDARD_ABS]) for _, row in df.iterrows()]
    return df, data_list


def plot_all(Bvals, Abs_meas, Abs_fit, A_free_arr, AB_arr, A2B_arr, FB_arr,
             Abs_err, title_tag, A3B_arr=None):
    # Plot Measured vs. Fitted ΔAbs
    plt.figure()
    if Abs_err is not None:
        plt.errorbar(Bvals, Abs_meas, yerr=Abs_err, fmt="o", label="Measured")
    else:
        plt.plot(Bvals, Abs_meas, "o", label="Measured")
    plt.plot(Bvals, Abs_fit, "-", label="Fit")
    plt.title(f"{title_tag}: Measured vs. Fit ΔAbs")
    plt.xlabel("[HSA] (M)")
    plt.ylabel("ΔAbs")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Fraction bound
    plt.figure()
    plt.plot(Bvals, FB_arr, "o-", label="Fraction Bound")
    plt.title(f"{title_tag}: Fraction Bound")
    plt.xlabel("[HSA] (M)")
    plt.ylabel("Fraction Bound")
    plt.tight_layout()
    plt.show()

    # Species plot
    plt.figure()
    plt.plot(Bvals, A_free_arr, "o-", label="[ICG_free]")
    plt.plot(Bvals, AB_arr, "o-", label="[HSA·ICG]")
    plt.plot(Bvals, A2B_arr, "o-", label="[HSA·ICG2]")
    if A3B_arr:
        plt.plot(Bvals, A3B_arr, "o-", label="[HSA·ICG3]")
    plt.title(f"{title_tag}: Species Concentrations")
    plt.xlabel("[HSA] (M)")
    plt.ylabel("Concentration (M)")
    plt.legend()
    plt.tight_layout()
    plt.show()
