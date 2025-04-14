#!/usr/bin/env python
"""
Three-Species Binding Model Fit

For the sequential binding:
    B + A <--> AB      (K1)
    AB + A <--> A2B    (K2)
    A2B + A <--> A3B   (K3)

Total ligand conservation leads to the cubic equation in a free ligand concentration a:
    (3 * B_T/(K1*K2*K3))*a^3 + (2 * B_T/(K1*K2))*a^2 + (1+B_T/K1)*a - A_T = 0

Once a is found, compute:
    [AB]   = a * B_T / K1
    [A2B]  = a^2 * B_T / (K1*K2)
    [A3B]  = a^3 * B_T / (K1*K2*K3)

We then assume the measured signal (e.g. ΔAbs) is linearly related to the bound fraction.
For example, one might define:
    fraction_bound = ([AB] + 2[A2B] + 3[A3B]) / A_T
and use:
    ΔAbs_model = offset + scale * fraction_bound

This script provides a complete pipeline for symbolic inspection,
numerical solution, parameter fitting using lmfit, and plotting.
"""

import sympy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, report_fit
import csv

from kinetics.config import INIT_PARAMS_3SPECIES

###############################################################################
# PART 1: SYMBOLIC DERIVATION (for inspection)
###############################################################################
def symbolic_derivation():
    """
    Derive the cubic binding equation symbolically and print its solutions.
    """
    # Define symbols
    a, K1, K2, K3, BT, AT = sympy.symbols("a K1 K2 K3 B_T A_T", positive=True)
    # Cubic equation in a (free ligand):
    # 3*B_T/(K1*K2*K3)*a^3 + 2*B_T/(K1*K2)*a^2 + (1+B_T/K1)*a - A_T = 0
    expr = (
        (3 * BT / (K1 * K2 * K3)) * a**3
        + (2 * BT / (K1 * K2)) * a**2
        + (1 + BT / K1) * a
        - AT
    )
    sol = sympy.solve(sympy.Eq(expr, 0), a, dict=True)
    print("=== Symbolic Solutions for free ligand a ===")
    for s in sol:
        print(s[a])


# Uncomment to run symbolic derivation for inspection:
# symbolic_derivation()


###############################################################################
# PART 2: Numerical Model Functions for 3-Species Binding
###############################################################################
def solve_free_A(AT, BT, K1, K2, K3):
    """
    Solve the cubic equation:
      (3 * BT/(K1*K2*K3))*a^3 + (2 * BT/(K1*K2))*a^2 + (1+B_T/K1)*a - AT = 0,
    and return the positive real root for a.
    """
    coeff3 = 3 * BT / (K1 * K2 * K3)
    coeff2 = 2 * BT / (K1 * K2)
    coeff1 = 1 + BT / K1
    coeff0 = -AT
    coeffs = [coeff3, coeff2, coeff1, coeff0]
    roots = np.roots(coeffs)
    real_roots = [r.real for r in roots if np.abs(r.imag) < 1e-8 and r.real > 0]
    if not real_roots:
        raise ValueError("No positive real root found!")
    return real_roots[0]


def calc_species_3(AT, BT, K1, K2, K3):
    """
    Given AT (total ligand), BT (total protein), and dissociation constants K1, K2, K3,
    compute:
         a_free  = [A_free]
         [AB]    = a_free*BT/K1
         [A2B]   = a_free^2 * BT/(K1*K2)
         [A3B]   = a_free^3 * BT/(K1*K2*K3)
    Returns a tuple: (a_free, AB, A2B, A3B)
    """
    a_free = solve_free_A(AT, BT, K1, K2, K3)
    AB = (a_free * BT) / K1
    A2B = (a_free**2 * BT) / (K1 * K2)
    A3B = (a_free**3 * BT) / (K1 * K2 * K3)
    return a_free, AB, A2B, A3B


def model_delta_abs_3(AT, BT, K1, K2, K3, offset, scale):
    """
    Compute the model-predicted ΔAbs.

    Here we define the bound fraction as:
      fraction_bound = ([AB] + 2*[A2B] + 3*[A3B]) / AT.
    Then, a simple linear mapping is used:
      ΔAbs_model = offset + scale * fraction_bound.

    Returns ΔAbs_model.
    """
    a_free, AB, A2B, A3B = calc_species_3(AT, BT, K1, K2, K3)
    frac_bound = (AB + 2.0 * A2B + 3.0 * A3B) / AT
    return offset + scale * frac_bound


###############################################################################
# PART 3: Fitting Routine for the 3-Species Model
###############################################################################
def objective(params, data_list):
    """
    For each data point (AT, BT, ΔAbs_meas), compute the residual between the
    model-predicted ΔAbs and the measured ΔAbs.

    The free parameters are K1, K2, K3, offset, and scale.
    """
    K1 = params["K1"].value
    K2 = params["K2"].value
    K3 = params["K3"].value
    offset = params["offset"].value
    scale = params["scale"].value

    residuals = []
    for AT, BT, deltaAbs_meas in data_list:
        deltaAbs_model = model_delta_abs_3(AT, BT, K1, K2, K3, offset, scale)
        residuals.append(deltaAbs_model - deltaAbs_meas)
    return np.array(residuals)


def fit_3species_deltaAbs(data_list):
    """
    Fit the three-species binding model to ΔAbs data.

    data_list is a list of tuples (AT, BT, ΔAbs_meas).

    Returns the lmfit result.
    """
    p = Parameters()
    for key, settings in INIT_PARAMS_3SPECIES.items():
        p.add(key, **settings)

    result = minimize(objective, p, args=(data_list,))
    # print("\n=== Fit Report ===")
    # print(report_fit(result))
    return result
