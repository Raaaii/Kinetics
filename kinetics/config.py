# kinetics/config.py

# === Data Settings ===
DEFAULT_DATA_FILE = "data/data.xlsx"
DEFAULT_SHEET_NAME = "Sheet1"

# === Initial Parameters for 2-Species Model ===
INIT_PARAMS_2SPECIES = {
    "K1": {"value": 1e-6, "min": 1e-7, "max": 1e-5},
    "K2": {"value": 1e-6, "min": 1e-7, "max": 1e-5},
    "offset": {"value": 2.1796e-04},
    "scale": {"value": 0.1},
}

# === Initial Parameters for 3-Species Model  ==
INIT_PARAMS_3SPECIES = {
    "K1": {"value": 1e-6, "min": 1e-10, "max": 1e2},
    "K2": {"value": 1e-6, "min": 1e-10, "max": 1e2},
    "K3": {"value": 1e-6, "min": 1e-10, "max": 1e2},
    "offset": {"value": 2.1796e-04},
    "scale": {"value": 0.1},
}

# config.py
# kinetics/config.py

# Raw column names in Excel
COLUMN_ICG = "A"
COLUMN_HSA = "B"
COLUMN_DABS = "m"
COLUMN_DABS_ERR = "Delta Abs Error"

# Standardized internal column names
STANDARD_LIGAND = "[ICG]"
STANDARD_PROTEIN = "[HSA]"
STANDARD_ABS = "Delta Abs"
