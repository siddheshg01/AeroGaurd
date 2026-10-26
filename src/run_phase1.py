"""AeroGuard AI - Phase 1 Pipeline Runner.

Executes data loading, schema assignment, validation, and exploratory data analysis.
"""

import sys
import time
from pathlib import Path

# Add project root to path if running directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import (
    TRAIN_FILE,
    TEST_FILE,
    RUL_FILE,
    FIGURES_DIR,
    REPORTS_DIR,
    ALL_COLUMNS,
)
from src.data_loader import load_all_data
from src.validator import validate_dataset
from src.eda import run_eda_pipeline


def print_banner():
    banner = """
========================================================================
                      AEROGUARD AI - PHASE 1
             C-MAPSS Ingestion, Validation & EDA Pipeline
========================================================================
"""
    print(banner)


def main():
    print_banner()
    start_time = time.time()

    # Step 1: Verify Dataset Path Integrity
    print("[1/5] Verifying input file paths...")
    for label, path in [("Train", TRAIN_FILE), ("Test", TEST_FILE), ("RUL", RUL_FILE)]:
        if not path.exists():
            print(f"  [ERROR] {label} file not found: {path}")
            sys.exit(1)
        size_kb = path.stat().st_size / 1024
        print(f"  [OK] Found {label} file: {path.name} ({size_kb:.1f} KB)")

    # Step 2: Load Datasets and Assign 26 Column Names
    print("\n[2/5] Loading datasets with 26 standard C-MAPSS columns...")
    try:
        train_df, test_df, rul_df = load_all_data()
        print(f"  [OK] Loaded train_FD001: {train_df.shape[0]:,} rows × {train_df.shape[1]} columns")
        print(f"  [OK] Loaded test_FD001:  {test_df.shape[0]:,} rows × {test_df.shape[1]} columns")
        print(f"  [OK] Loaded RUL_FD001:   {rul_df.shape[0]} rows × {rul_df.shape[1]} columns")
        print(f"  [OK] Schema: 2 identifiers, 3 operational settings, 21 sensor measurements")
    except Exception as e:
        print(f"  [ERROR] Failed during data loading: {e}")
        sys.exit(1)

    # Step 3: Run Validation Suite
    print("\n[3/5] Performing data validation checks...")
    is_valid, validation_report = validate_dataset(train_df, test_df, rul_df)

    if not is_valid:
        print("  [FAILED] Data validation detected integrity failures:")
        for failure in validation_report.get("failures", []):
            print(f"    - {failure}")
        sys.exit(1)

    train_stats = validation_report["cycle_ranges"]["train_max_cycle_stats"]
    print("  [PASS] Shapes and column counts validated (26 columns)")
    print(f"  [PASS] Unique engines verified: {validation_report['engine_counts']['train_unique_engines']} Train, {validation_report['engine_counts']['test_unique_engines']} Test")
    print(f"  [PASS] Zero missing values (NaNs) across all {train_df.shape[1]} columns")
    print(f"  [PASS] Zero duplicate rows / keys verified")
    print(f"  [PASS] Cycle range verified (All engines start at cycle 1)")
    print(f"         Engine lifespan (Train): Min={train_stats['min']} | Mean={train_stats['mean']} | Median={train_stats['median']} | Max={train_stats['max']} cycles")
    print(f"  [OK] Reports saved to: {REPORTS_DIR}")

    # Step 4: Run Exploratory Data Analysis & Visualizations
    print("\n[4/5] Running Exploratory Data Analysis & Generating Visualizations...")
    eda_results = run_eda_pipeline(train_df)
    print(f"  [OK] Computed sensor variance and descriptive statistics")
    print(f"       - Informative/Active sensors: {len(eda_results['informative_sensors'])} sensors")
    print(f"       - Invariant/Constant sensors: {len(eda_results['constant_sensors'])} sensors {eda_results['constant_sensors']}")
    print(f"  [OK] Generated: {eda_results['trajectory_plot']}")
    print(f"  [OK] Generated: {eda_results['lifespan_plot']}")
    print(f"  [OK] Generated: {eda_results['correlation_plot']}")
    print(f"  [OK] Generated: {eda_results['interactive_html']}")

    # Step 5: Summary
    elapsed = time.time() - start_time
    print("\n[5/5] Phase 1 Execution Summary:")
    print("------------------------------------------------------------------------")
    print(f"Status:             ALL PHASE 1 CHECKS PASSED")
    print(f"Execution Time:     {elapsed:.2f} seconds")
    print(f"Reports Location:   {REPORTS_DIR}")
    print(f"Figures Location:   {FIGURES_DIR}")
    print("------------------------------------------------------------------------")
    print("Phase 1 complete. Dataset is validated, characterized, and ready for Phase 2.")


if __name__ == "__main__":
    main()
