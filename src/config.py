"""Configuration and Constants for AeroGuard AI Phase 1."""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "CMAPSSData"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Dataset File Paths
TRAIN_FILE = DATA_DIR / "train_FD001.txt"
TEST_FILE = DATA_DIR / "test_FD001.txt"
RUL_FILE = DATA_DIR / "RUL_FD001.txt"

# Exact 26 Column Names for C-MAPSS FD001:
# 1) unit number
# 2) cycle
# 3-5) three operational settings
# 6-26) 21 sensor measurements
INDEX_COLUMNS = ["unit_number", "cycle"]
SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
FEATURE_COLUMNS = SETTING_COLUMNS + SENSOR_COLUMNS
ALL_COLUMNS = INDEX_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS

# Sensor Metadata Descriptions (from NASA C-MAPSS technical documentation)
SENSOR_DESCRIPTIONS = {
    "sensor_1": "Fan inlet temperature (T2) [°R]",
    "sensor_2": "LPC outlet temperature (T24) [°R]",
    "sensor_3": "HPC outlet temperature (T30) [°R]",
    "sensor_4": "LPT outlet temperature (T50) [°R]",
    "sensor_5": "Fan inlet pressure (P2) [psia]",
    "sensor_6": "Bypass-duct pressure (P15) [psia]",
    "sensor_7": "HPC outlet pressure (P30) [psia]",
    "sensor_8": "Physical fan speed (Nf) [rpm]",
    "sensor_9": "Physical core speed (Nc) [rpm]",
    "sensor_10": "Engine pressure ratio (epr) [--]",
    "sensor_11": "HPC static pressure (Ps30) [psia]",
    "sensor_12": "Ratio of fuel flow to Ps30 (phi) [pps/psi]",
    "sensor_13": "Corrected fan speed (NRf) [rpm]",
    "sensor_14": "Corrected core speed (NRc) [rpm]",
    "sensor_15": "Bypass ratio (BPR) [--]",
    "sensor_16": "Burner fuel-air ratio (farB) [--]",
    "sensor_17": "Bleed enthalpy (htBleed) [--]",
    "sensor_18": "Demanded fan speed (Nf_dmd) [rpm]",
    "sensor_19": "Demanded corrected fan speed (PCNfR_dmd) [rpm]",
    "sensor_20": "HPT coolant bleed (W31) [lbm/s]",
    "sensor_21": "LPT coolant bleed (W32) [lbm/s]",
}

# Expected Dataset Properties for FD001
EXPECTED_COLUMNS_COUNT = 26
EXPECTED_TRAIN_UNITS = 100
EXPECTED_TEST_UNITS = 100
EXPECTED_RUL_COUNT = 100
