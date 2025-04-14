# run.py (updated version with import fix)

import argparse
import pandas as pd
from kinetics.utils import print_fit_params, summarize_results

# Fix for relative import issue when running as script
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent))

from kinetics.two_species import (
    fit_2step_deltaAbs, model_delta_abs, calc_species_concentrations
)
from kinetics.three_species import (
    fit_3species_deltaAbs, model_delta_abs_3, calc_species_3
)
from kinetics.utils import load_data_and_build_list, plot_all

from kinetics.config import DEFAULT_DATA_FILE, DEFAULT_SHEET_NAME



def main():
    parser = argparse.ArgumentParser(description="Fit binding model to absorbance data")
    parser.add_argument("model", choices=["2species", "3species"], help="Choose model to fit")
    parser.add_argument("--file", default=DEFAULT_DATA_FILE, help="Path to Excel data file")
    parser.add_argument("--sheet", default=DEFAULT_SHEET_NAME, help="Excel sheet name")
    args = parser.parse_args()

    # Load data
    df, data_list = load_data_and_build_list(args.file, args.sheet)
    Bvals = df["[HSA]"].values
    Abs_meas = df["Delta Abs"].values
    Abs_err = df["Delta Abs Error"].values if "Delta Abs Error" in df.columns else None

    if args.model == "2species":
        result = fit_2step_deltaAbs(data_list)
        K1 = result.params["K1"].value
        K2 = result.params["K2"].value
        offset = result.params["offset"].value
        scale = result.params["scale"].value

        Abs_fit, A_free_arr, AB_arr, A2B_arr, FB_arr = [], [], [], [], []
        for AT, BT, _ in data_list:
            Abs_fit.append(model_delta_abs(AT, BT, K1, K2, offset, scale))
            a_free, AB, A2B, fb = calc_species_concentrations(AT, BT, K1, K2)
            A_free_arr.append(a_free)
            AB_arr.append(AB)
            A2B_arr.append(A2B)
            FB_arr.append(fb)

        plot_all(Bvals, Abs_meas, Abs_fit, A_free_arr, AB_arr, A2B_arr, FB_arr, Abs_err, "2-Species")

    elif args.model == "3species":
        result = fit_3species_deltaAbs(data_list)
        K1 = result.params["K1"].value
        K2 = result.params["K2"].value
        K3 = result.params["K3"].value
        offset = result.params["offset"].value
        scale = result.params["scale"].value

        Abs_fit, A_free_arr, AB_arr, A2B_arr, A3B_arr, FB_arr = [], [], [], [], [], []
        for AT, BT, _ in data_list:
            Abs_fit.append(model_delta_abs_3(AT, BT, K1, K2, K3, offset, scale))
            a_free, AB, A2B, A3B = calc_species_3(AT, BT, K1, K2, K3)
            A_free_arr.append(a_free)
            AB_arr.append(AB)
            A2B_arr.append(A2B)
            A3B_arr.append(A3B)
            FB_arr.append((AB + 2*A2B + 3*A3B) / AT)

        plot_all(Bvals, Abs_meas, Abs_fit, A_free_arr, AB_arr, A2B_arr, FB_arr, Abs_err, "3-Species", A3B_arr=A3B_arr)
        # After fitting



    print_fit_params(result.params, model="2-Species" if args.model == "2species" else "3-Species")

    save_name = "two_species_fit.csv" if args.model == "2species" else "three_species_fit.csv"

    results_df = summarize_results(
        model_name="2-Species" if args.model == "2species" else "3-Species",
        Bvals=Bvals,
        Abs_meas=Abs_meas,
        Abs_fit=Abs_fit,
        A_free_arr=A_free_arr,
        AB_arr=AB_arr,
        A2B_arr=A2B_arr,
        A3B_arr=A3B_arr if args.model == "3species" else None,
        FB_arr=FB_arr,
        save_path=f"results/{save_name}"
    )
main()