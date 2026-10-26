# AeroGuard AI - Phase 1 Data Validation Report

**Status:** `PASS`

## 1. Dataset Dimensions & Schema
- **Training Set (train_FD001):** 20,631 rows × 26 columns
- **Testing Set (test_FD001):** 13,096 rows × 26 columns
- **Ground Truth RUL (RUL_FD001):** 100 rows × 2 columns
- **Column Schema:** 26 assigned columns (unit_number, cycle, 3 operational settings, 21 sensor measurements).

## 2. Fleet Engine Counts
- **Training Engines:** 100 (IDs 1 to 100)
- **Testing Engines:** 100 (IDs 1 to 100)
- **RUL Engine Labels:** 100 ground-truth values

## 3. Data Hygiene & Integrity
- **Missing Values (NaNs):**
  - Train: 0
  - Test: 0
  - RUL: 0
- **Duplicate Rows:**
  - Full Row Duplicates (Train / Test): 0 / 0
  - Primary Key (unit, cycle) Collisions: 0 / 0

## 4. Cycle Range & Monotonicity
- All engines start at Cycle 1: Train = `True`, Test = `True`
- Cycles are strictly monotonic: Train = `True`, Test = `True`
- **Train Run-to-Failure Lifespan (Cycles):**
  - Min: 128 cycles
  - Max: 362 cycles
  - Mean: 206.31 cycles
  - Median: 199.0 cycles
  - Standard Deviation: 46.34 cycles
- **Test Cut-off Snapshot (Cycles):**
  - Min: 31 cycles
  - Max: 303 cycles
  - Mean: 130.96 cycles
  - Median: 133.5 cycles

## 5. Result
**Total Failures:** 0

All Phase 1 data validation checks passed successfully with 0 defects.
