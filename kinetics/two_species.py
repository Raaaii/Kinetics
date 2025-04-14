#!/usr/bin/env python
"""
Two-Step Binding Analysis from Absorbance Data

This script fits a two-step binding model to experimental absorbance data.
The experiment measures a change in absorbance (ΔAbs) at λ=800 nm as [HSA]
(varied total protein) is titrated into a fixed [ICG] (ligand).

The binding scheme is assumed to be:
    HSA + ICG <--> HSA·ICG      (dissociation constant K1)
    HSA·ICG + ICG <--> HSA·ICG2   (dissociation constant K2)

Using mass conservation, the fraction of ICG bound is calculated by solving a
quadratic for [ICG_free] (denoted as a_free). Then the complexes are computed:
    [HSA·ICG]  = [ICG_free]·[HSA] / K1,
    [HSA·ICG2] = ([ICG_free]²·[HSA]) / (K1·K2).

The total fraction bound is then:
    f_bound = ([HSA·ICG] + 2·[HSA·ICG2]) / [ICG]_total

We assume that the measured ΔAbs is linearly related to f_bound:
    ΔAbs_model = offset + scale * f_bound

The fitting parameters are thus K1, K2, offset, and scale.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, report_fit

from kinetics.config import INIT_PARAMS_2SPECIES


###############################################################################
# PART 1: Core Functions (Model and Calculations)
###############################################################################
def solve_free_A_2step(AT, BT, K1, K2):
    """
    Solve for the free ligand concentration [A_free] (ICG_free) in the two‐step model.

    The total ICG conservation yields a quadratic:
         alpha * a² + beta * a - AT = 0,
    with:
         alpha = 2 * BT / (K1 * K2),
         beta  = 1 + BT / K1,
         a = [A_free].

    Parameters
    ----------
    AT : float
        Total ICG (ligand) concentration (M).
    BT : float
        Total HSA (protein) concentration (M).
    K1 : float
        Dissociation constant for the first binding step (M).
    K2 : float
        Dissociation constant for the second binding step (M).

    Returns
    -------
    a_free : float
        The free ICG concentration (M).

    Raises
    ------
    ValueError
        If no positive real root is found.
    """
    alpha = 2.0 * BT / (K1 * K2)
    beta = 1.0 + (BT / K1)
    gamma = -AT  # -Total ICG

    coeffs = [alpha, beta, gamma]
    roots = np.roots(coeffs)

    # Choose the positive real solution.
    real_pos = [r.real for r in roots if abs(r.imag) < 1e-12 and r.real > 0]
    # print("The positive roots are", real_pos)

    if not real_pos:
        raise ValueError(f"No positive root for AT={AT}, BT={BT}, (K1={K1}, K2={K2}).")
    return real_pos[0]


def model_delta_abs(AT, BT, K1, K2, offset, scale):
    """
    Calculate the model-predicted ΔAbs.

    Steps:
      1. Solve for free ICG: a_free = [ICG_free].
      2. Calculate [HSA·ICG]  = (a_free * BT) / K1.
      3. Calculate [HSA·ICG2] = (a_free² * BT) / (K1 * K2).
      4. Compute fraction bound: f_bound = ([HSA·ICG] + 2*[HSA·ICG2]) / AT.
      5. Map fraction bound to ΔAbs via: ΔAbs = offset + scale * f_bound.

    Parameters
    ----------
    AT : float
        Total ICG concentration (M).
    BT : float
        Total HSA concentration (M).
    K1, K2 : float
        Dissociation constants (M).
    offset : float
        Baseline ΔAbs.
    scale : float
        Scaling factor from fraction bound to ΔAbs.

    Returns
    -------
    delta_abs : float
        Model-predicted ΔAbs.
    """
    a_free = solve_free_A_2step(AT, BT, K1, K2)
    AB = (a_free * BT) / K1
    A2B = (a_free**2 * BT) / (K1 * K2)

    frac_bound = (AB + 2.0 * A2B) / AT
    return offset + scale * frac_bound


def calc_species_concentrations(AT, BT, K1, K2):
    """
    Compute the species concentrations for given AT, BT, and best-fit K1, K2.

    Returns:
        a_free: [ICG_free] (M)
        AB: [HSA·ICG] (M)
        A2B: [HSA·ICG2] (M)
        frac_bound: fraction of ICG bound = ([HSA·ICG] + 2*[HSA·ICG2]) / AT
    """
    a_free = solve_free_A_2step(AT, BT, K1, K2)
    AB = (a_free * BT) / K1
    A2B = (a_free**2 * BT) / (K1 * K2)
    frac_bound = (AB + 2 * A2B) / AT
    return a_free, AB, A2B, frac_bound



###############################################################################
# PART 2: Fitting Functions
###############################################################################
def objective(params, data_list):
    """
    Compute the residuals between model and measured ΔAbs.

    Parameters
    ----------
    params : lmfit.Parameters
        Contains the free parameters K1, K2, offset, scale.
    data_list : list of tuples
        Each tuple is (AT, BT, deltaAbs_meas).

    Returns
    -------
    residuals : np.array
        Array of differences: (ΔAbs_model - ΔAbs_meas).
    """
    K1 = params["K1"].value
    K2 = params["K2"].value
    offset = params["offset"].value
    scale = params["scale"].value

    residuals = []
    for AT, BT, deltaAbs_meas in data_list:
        deltaAbs_pred = model_delta_abs(AT, BT, K1, K2, offset, scale)
        residuals.append(deltaAbs_pred - deltaAbs_meas)
    return np.array(residuals)


def fit_2step_deltaAbs(data_list):
    """
    Fit the two-step binding model to ΔAbs data.

    data_list : list of (AT, BT, deltaAbs_meas) tuples.

    Returns
    -------
    result : lmfit.MinimizerResult
        Contains the best-fit parameters and fit report.
    """
    # Set initial guesses (tweak as needed)
    p = Parameters()
    p = Parameters()
    for key, settings in INIT_PARAMS_2SPECIES.items():
        p.add(key, **settings)

    result = minimize(objective, p, args=(data_list,))
    # print("\n=== Fit Report ===")
    # print(report_fit(result))
    return result

