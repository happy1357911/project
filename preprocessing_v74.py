from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional
from zipfile import ZipFile

import numpy as np
import pandas as pd

from clinical_rules_v74 import (
    MISSING_TOKENS,
    TASK2_ABG_FREQS,
    TASK2_AC_NR_LIMITS_DB,
    TASK2_BC_NR_LIMITS_DB,
    TASK2_EAR_LEVEL_AC_FREQS,
    TASK2_STANDARD_HEARING_TYPES,
    TASK3_NAME_MAP,
    TASK3_NP_EXTREME_VALUES,
    TASK3_SIDE_NP_EXTREME_VALUES,
    clean_value_none,
    clean_value_zero,
    hz_from_col,
    infer_task2_hearing_type_from_rule,
    is_nr_value,
    mean_present,
    nr_indicator_series,
    numeric_series,
    parse_numeric_value,
    predict_task3_type_from_rule,
    rule_decision as shared_rule_decision,
    task2_evidence_decision,
)


SIDES = ["right", "left"]
INSUFFICIENT_EVIDENCE_LABEL = "INSUFFICIENT_EVIDENCE"

TASK1_EAR_LEVEL_FEATURES = [
    "ac_500Hz",
    "ac_1000Hz",
    "ac_2000Hz",
    "ac_4000Hz",
    "ac_PTA",
]

TASK2_EAR_LEVEL_FEATURES = [
    "ac_500Hz",
    "ac_1000Hz",
    "ac_2000Hz",
    "ac_4000Hz",
    "ac_500Hz_nr",
    "ac_1000Hz_nr",
    "ac_2000Hz_nr",
    "ac_4000Hz_nr",
    "bc_500Hz",
    "bc_1000Hz",
    "bc_2000Hz",
    "bc_4000Hz",
    "bc_500Hz_nr",
    "bc_1000Hz_nr",
    "bc_2000Hz_nr",
    "bc_4000Hz_nr",
    "bc_500Hz_missing",
    "bc_1000Hz_missing",
    "bc_2000Hz_missing",
    "bc_4000Hz_missing",
    "abg_500Hz",
    "abg_1000Hz",
    "abg_2000Hz",
    "abg_4000Hz",
    "abg_500Hz_missing",
    "abg_1000Hz_missing",
    "abg_2000Hz_missing",
    "abg_4000Hz_missing",
    "abg_500Hz_censored",
    "abg_1000Hz_censored",
    "abg_2000Hz_censored",
    "abg_4000Hz_censored",
]

TASK3_BASE_FEATURES = [
    "tymp_right_Vea",
    "tymp_right_peak_daPa",
    "tymp_right_peak_mmho",
    "tymp_right_Width_daPa",
    "tymp_left_Vea",
    "tymp_left_peak_daPa",
    "tymp_left_peak_mmho",
    "tymp_left_Width_daPa",
]

TASK3_EAR_LEVEL_BASE_FEATURES = [
    "tymp_Vea",
    "tymp_peak_daPa",
    "tymp_peak_mmho",
    "tymp_Width_daPa",
]

TASK3_EAR_LEVEL_DERIVED_FEATURES = []
for _col in TASK3_EAR_LEVEL_BASE_FEATURES:
    TASK3_EAR_LEVEL_DERIVED_FEATURES.extend([
        f"{_col}_real_zero",
        f"{_col}_missing_zero",
        f"{_col}_np_zero",
    ])

TASK3_EAR_LEVEL_FEATURES = [
    *TASK3_EAR_LEVEL_BASE_FEATURES,
    *TASK3_EAR_LEVEL_DERIVED_FEATURES,
]

TASK_INFO = {
    "Task1": {
        "csv": "task1_all_three_common14_v1.csv",
        "label_cols": ["hearing_degree_WHO_PTAbased"],
        "feature_cols": TASK1_EAR_LEVEL_FEATURES,
    },
    "Task2": {
        "csv": "task2_3_pure_data.csv",
        "label_cols": ["hearing_type"],
        "feature_cols": TASK2_EAR_LEVEL_FEATURES,
    },
    "Task3": {
        "csv": "task2_3_pure_data.csv",
        "label_cols": ["tymp_type"],
        "feature_cols": TASK3_EAR_LEVEL_FEATURES,
    },
}



XLSX_SIGNATURE = b"PK\x03\x04"
SPREADSHEET_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
RELATIONSHIP_NS = {"rel": "http://schemas.openxmlformats.org/package/2006/relationships"}


def _xlsx_col_index(cell_ref: str) -> int:
    letters = re.sub(r"[^A-Z]", "", str(cell_ref).upper())
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return max(idx - 1, 0)


def _xlsx_cell_value(cell, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    value = cell.find("a:v", SPREADSHEET_NS)
    inline = cell.find("a:is", SPREADSHEET_NS)
    if cell_type == "s" and value is not None and value.text is not None:
        return shared_strings[int(value.text)]
    if cell_type == "inlineStr" and inline is not None:
        return "".join(text.text or "" for text in inline.findall(".//a:t", SPREADSHEET_NS))
    if value is not None and value.text is not None:
        return value.text
    return ""


def _first_xlsx_sheet_path(zf: ZipFile) -> str:
    sheet_path = "xl/worksheets/sheet1.xml"
    try:
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_by_id = {rel.attrib.get("Id"): rel.attrib.get("Target") for rel in rels.findall("rel:Relationship", RELATIONSHIP_NS)}
        first_sheet = workbook.find("a:sheets/a:sheet", SPREADSHEET_NS)
        if first_sheet is not None:
            rid = first_sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
            target = rel_by_id.get(rid)
            if target:
                target = target.lstrip("/")
                sheet_path = target if target.startswith("xl/") else f"xl/{target}"
    except Exception:
        pass
    return sheet_path


def _read_xlsx_first_sheet(path: Path) -> pd.DataFrame:
    with ZipFile(path) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            shared_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for item in shared_root.findall("a:si", SPREADSHEET_NS):
                shared_strings.append("".join(text.text or "" for text in item.findall(".//a:t", SPREADSHEET_NS)))

        sheet_path = _first_xlsx_sheet_path(zf)
        sheet_root = ET.fromstring(zf.read(sheet_path))
        rows: list[list[str]] = []
        for row in sheet_root.findall(".//a:sheetData/a:row", SPREADSHEET_NS):
            cells: dict[int, str] = {}
            max_idx = -1
            for cell in row.findall("a:c", SPREADSHEET_NS):
                ref = cell.attrib.get("r", "")
                idx = _xlsx_col_index(ref) if ref else len(cells)
                cells[idx] = _xlsx_cell_value(cell, shared_strings)
                max_idx = max(max_idx, idx)
            if max_idx >= 0:
                values = [""] * (max_idx + 1)
                for idx, value in cells.items():
                    values[idx] = value
                rows.append(values)

    if not rows:
        return pd.DataFrame()
    header = [str(value).strip() for value in rows[0]]
    while header and header[-1] == "":
        header.pop()
    normalized_rows = []
    for row in rows[1:]:
        row = row[:len(header)] + [""] * max(0, len(header) - len(row))
        normalized_rows.append(row[:len(header)])
    return pd.DataFrame(normalized_rows, columns=header)


def load_tabular_data(path: str | Path) -> pd.DataFrame:
    """Load normal CSV files and xlsx files whose extension may still be .csv."""
    path = Path(path)
    with path.open("rb") as fh:
        signature = fh.read(4)
    if signature == XLSX_SIGNATURE:
        return _read_xlsx_first_sheet(path)
    return pd.read_csv(path)

def sanitize_column_name(value: str) -> str:
    return re.sub(r"\s+", "_", str(value).strip())


def normalize_side(side) -> str:
    side = str(side).strip().lower()
    return side if side in set(SIDES) else "right"


def col_or_nan(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series(np.nan, index=df.index)


def ensure_case_id(df: pd.DataFrame) -> pd.DataFrame:
    if "case_id" not in df.columns:
        df = df.copy()
        df["case_id"] = np.arange(len(df))
    return df


def flag_value(row: Mapping, col: str, fallback: float = 0.0) -> float:
    value = row.get(col, fallback)
    parsed = parse_numeric_value(value)
    if parsed is None:
        return float(fallback)
    return 1.0 if parsed >= 0.5 else 0.0


def row_has(row: Mapping, col: str) -> bool:
    if col not in row:
        return False
    value = row.get(col)
    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass
    return str(value).strip().lower() not in {token.lower() for token in MISSING_TOKENS}


def row_numeric(row: Mapping, col: str, nr_limits: Optional[Dict[int, float]] = None):
    return parse_numeric_value(row.get(col), hz=hz_from_col(col), nr_limits=nr_limits)


def first_numeric(row: Mapping, cols: Iterable[str], nr_limits: Optional[Dict[int, float]] = None):
    for col in cols:
        if col in row:
            value = row_numeric(row, col, nr_limits=nr_limits)
            if value is not None:
                return value
    return None


def first_nr_flag(row: Mapping, cols: Iterable[str]) -> float:
    for col in cols:
        if col in row and is_nr_value(row.get(col)):
            return 1.0
    return 0.0


def convert_task2_nr_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for side in SIDES:
        for hz in TASK2_AC_NR_LIMITS_DB:
            ac_col = f"{side}_{hz}Hz"
            if ac_col in df.columns:
                df[f"{ac_col}_nr"] = nr_indicator_series(df, ac_col)
                df[ac_col] = numeric_series(df, ac_col, TASK2_AC_NR_LIMITS_DB)
        for hz in TASK2_ABG_FREQS:
            bc_col = f"bc_{side}_{hz}Hz"
            if bc_col in df.columns:
                df[f"{bc_col}_nr"] = nr_indicator_series(df, bc_col)
                df[bc_col] = numeric_series(df, bc_col, TASK2_BC_NR_LIMITS_DB)
    return df


def add_task2_measurement_features(df: pd.DataFrame) -> pd.DataFrame:
    derived_cols = {}
    for side in SIDES:
        for hz in TASK2_ABG_FREQS:
            ac_col = f"{side}_{hz}Hz"
            bc_col = f"bc_{side}_{hz}Hz"
            ac = numeric_series(df, ac_col, TASK2_AC_NR_LIMITS_DB)
            bc = numeric_series(df, bc_col, TASK2_BC_NR_LIMITS_DB)
            ac_nr = pd.to_numeric(df.get(f"{ac_col}_nr", 0.0), errors="coerce").fillna(0.0)
            bc_nr = pd.to_numeric(df.get(f"{bc_col}_nr", 0.0), errors="coerce").fillna(0.0)

            derived_cols[f"{bc_col}_missing"] = bc.isna().astype(float)
            derived_cols[f"abg_{side}_{hz}Hz"] = ac - bc
            derived_cols[f"abg_{side}_{hz}Hz_missing"] = (ac.isna() | bc.isna()).astype(float)
            abg_censored = ((ac_nr >= 0.5) | (bc_nr >= 0.5)) & ~(ac.isna() | bc.isna())
            derived_cols[f"abg_{side}_{hz}Hz_censored"] = abg_censored.astype(float)
    return pd.concat([df, pd.DataFrame(derived_cols, index=df.index)], axis=1)


def build_task1_ear_level_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    ear_frames = []
    for side in SIDES:
        ear_df = pd.DataFrame(index=df.index)
        ear_df["case_id"] = col_or_nan(df, "case_id")
        ear_df["ear_side"] = side
        ear_df["hearing_degree_WHO_PTAbased"] = col_or_nan(df, f"hearing_degree_WHO_PTAbased_{side}")
        for hz in [500, 1000, 2000, 4000]:
            ear_df[f"ac_{hz}Hz"] = col_or_nan(df, f"{side}_{hz}Hz")
        ear_df["ac_PTA"] = col_or_nan(df, f"{side}_PTA")
        ear_frames.append(ear_df)
    return pd.concat(ear_frames, axis=0, ignore_index=True)


def build_task2_ear_level_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    ear_frames = []
    for side in SIDES:
        ear_df = pd.DataFrame(index=df.index)
        ear_df["case_id"] = col_or_nan(df, "case_id")
        ear_df["ear_side"] = side
        ear_df["hearing_type"] = col_or_nan(df, f"hearing_type_{side}")
        for hz in TASK2_EAR_LEVEL_AC_FREQS:
            ear_df[f"ac_{hz}Hz"] = col_or_nan(df, f"{side}_{hz}Hz")
            ear_df[f"ac_{hz}Hz_nr"] = col_or_nan(df, f"{side}_{hz}Hz_nr")
        for hz in TASK2_ABG_FREQS:
            ear_df[f"bc_{hz}Hz"] = col_or_nan(df, f"bc_{side}_{hz}Hz")
            ear_df[f"bc_{hz}Hz_nr"] = col_or_nan(df, f"bc_{side}_{hz}Hz_nr")
            ear_df[f"bc_{hz}Hz_missing"] = col_or_nan(df, f"bc_{side}_{hz}Hz_missing")
            ear_df[f"abg_{hz}Hz"] = col_or_nan(df, f"abg_{side}_{hz}Hz")
            ear_df[f"abg_{hz}Hz_missing"] = col_or_nan(df, f"abg_{side}_{hz}Hz_missing")
            ear_df[f"abg_{hz}Hz_censored"] = col_or_nan(df, f"abg_{side}_{hz}Hz_censored")
        ear_frames.append(ear_df)
    return pd.concat(ear_frames, axis=0, ignore_index=True)


def build_task3_ear_level_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    ear_frames = []
    for side in SIDES:
        ear_df = pd.DataFrame(index=df.index)
        ear_df["case_id"] = col_or_nan(df, "case_id")
        ear_df["ear_side"] = side
        ear_df["tymp_type"] = col_or_nan(df, f"tymp_{side}_type")
        for suffix, neutral_col in TASK3_NAME_MAP.items():
            src_col = f"tymp_{side}_{suffix}"
            ear_df[neutral_col] = col_or_nan(df, src_col)
            for flag in ["real_zero", "missing_zero", "np_zero"]:
                ear_df[f"{neutral_col}_{flag}"] = col_or_nan(df, f"{src_col}_{flag}")
        ear_frames.append(ear_df)
    return pd.concat(ear_frames, axis=0, ignore_index=True)


def filter_task2_standard_hearing_types(df: pd.DataFrame) -> pd.DataFrame:
    if "hearing_type" not in df.columns:
        return df
    label = df["hearing_type"].astype(str).str.strip().str.upper()
    keep = label.isin(TASK2_STANDARD_HEARING_TYPES)
    out = df.loc[keep].reset_index(drop=True)
    out.attrs["task2_label_filter"] = {
        "allowed_labels": sorted(TASK2_STANDARD_HEARING_TYPES),
        "rows_before_filter": int(len(df)),
        "rows_after_filter": int(len(out)),
        "rows_excluded": int((~keep).sum()),
        "labels_before_filter": {
            str(k): int(v)
            for k, v in label.value_counts(dropna=False).sort_index().items()
        },
        "excluded_labels": {
            str(k): int(v)
            for k, v in label.loc[~keep].value_counts(dropna=False).sort_index().items()
        },
    }
    return out


def prepare_task1_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [sanitize_column_name(col) for col in df.columns]
    return build_task1_ear_level_dataframe(ensure_case_id(df))


def prepare_task2_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [sanitize_column_name(col) for col in df.columns]
    df = ensure_case_id(df)
    df = convert_task2_nr_columns(df)
    df = add_task2_measurement_features(df)
    df = build_task2_ear_level_dataframe(df)
    return filter_task2_standard_hearing_types(df)


def prepare_task3_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [sanitize_column_name(col) for col in df.columns]
    df = ensure_case_id(df)

    for col in TASK3_BASE_FEATURES:
        if col in df.columns:
            raw = df[col].astype(str).str.strip()
            numeric = pd.to_numeric(df[col], errors="coerce")
        else:
            raw = pd.Series("", index=df.index, dtype=str)
            numeric = pd.Series(np.nan, index=df.index, dtype=float)

        is_np = raw.str.upper().eq("NP")
        is_missing = raw.isin(MISSING_TOKENS) | raw.isna()
        is_real_zero = numeric.eq(0.0) & ~is_missing & ~is_np

        df[f"{col}_real_zero"] = is_real_zero.astype(float)
        df[f"{col}_missing_zero"] = is_missing.astype(float)
        df[f"{col}_np_zero"] = is_np.astype(float)

        if col in TASK3_SIDE_NP_EXTREME_VALUES:
            df.loc[is_np, col] = TASK3_SIDE_NP_EXTREME_VALUES[col]

    return build_task3_ear_level_dataframe(df)


def prepare_task_dataframe(task_name: str, raw: pd.DataFrame) -> pd.DataFrame:
    if task_name == "Task1":
        return prepare_task1_dataframe(raw)
    if task_name == "Task2":
        return prepare_task2_dataframe(raw)
    if task_name == "Task3":
        return prepare_task3_dataframe(raw)
    raise KeyError(f"Unknown task: {task_name}")


def clean_label_columns(df: pd.DataFrame, label_cols: list[str]):
    out = df.copy()
    missing_stats = {}
    for lbl in label_cols:
        if lbl not in out.columns:
            raise KeyError(f"Missing label column: {lbl}")
        s = out[lbl].astype(str).str.strip()
        missing_lookup = {str(token).strip().lower() for token in MISSING_TOKENS}
        s = s.mask(s.str.lower().isin(missing_lookup), np.nan)
        out[lbl] = s
        missing_stats[lbl] = int(out[lbl].isna().sum())

    before = len(out)
    out = out.dropna(subset=label_cols).reset_index(drop=True)
    missing_stats["rows_before"] = int(before)
    missing_stats["rows_after"] = int(len(out))
    missing_stats["rows_dropped_any_label_missing"] = int(before - len(out))
    return out, missing_stats


def pad_and_clean(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out.columns = [sanitize_column_name(c) for c in out.columns]
    for col in feature_cols:
        if col not in out.columns:
            out[col] = 0.0
    for col in feature_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def task3_flags(raw_value):
    raw = str(raw_value).strip()
    numeric = pd.to_numeric(pd.Series([raw_value]), errors="coerce").iloc[0]
    is_np = raw.upper() == "NP"
    is_missing = raw.lower() in {token.lower() for token in MISSING_TOKENS} or pd.isna(raw_value)
    is_real_zero = bool(pd.notna(numeric) and float(numeric) == 0.0 and not is_missing and not is_np)
    return float(is_real_zero), float(is_missing), float(is_np)


def build_task1_model_row(row: Mapping, side: str) -> dict:
    out = dict(row)
    side = normalize_side(side)
    out["ear_side"] = side
    for hz in [500, 1000, 2000, 4000]:
        value = first_numeric(row, [f"ac_{hz}Hz", f"{side}_{hz}Hz"], TASK2_AC_NR_LIMITS_DB)
        out[f"ac_{hz}Hz"] = 0.0 if value is None else value
    pta = first_numeric(row, ["ac_PTA", f"{side}_PTA"], TASK2_AC_NR_LIMITS_DB)
    out["ac_PTA"] = 0.0 if pta is None else pta
    return out


def build_task2_model_row(row: Mapping, side: str) -> dict:
    out = build_task1_model_row(row, side)
    side = normalize_side(side)
    bc_values = []
    abg_values = []
    abg_censored_values = []

    for hz in TASK2_EAR_LEVEL_AC_FREQS:
        ac_cols = [f"ac_{hz}Hz", f"{side}_{hz}Hz"]
        value = first_numeric(row, ac_cols, TASK2_AC_NR_LIMITS_DB)
        out[f"ac_{hz}Hz"] = 0.0 if value is None else value
        out[f"ac_{hz}Hz_nr"] = flag_value(row, f"ac_{hz}Hz_nr", first_nr_flag(row, ac_cols))

    for hz in TASK2_ABG_FREQS:
        ac_cols = [f"ac_{hz}Hz", f"{side}_{hz}Hz"]
        bc_cols = [f"bc_{hz}Hz", f"bc_{side}_{hz}Hz"]
        ac = first_numeric(row, ac_cols, TASK2_AC_NR_LIMITS_DB)
        bc = first_numeric(row, bc_cols, TASK2_BC_NR_LIMITS_DB)
        bc_flag_col = f"bc_{hz}Hz_missing"
        abg_flag_col = f"abg_{hz}Hz_missing"
        abg_censored_col = f"abg_{hz}Hz_censored"
        ac_nr = flag_value(row, f"ac_{hz}Hz_nr", first_nr_flag(row, ac_cols))
        bc_nr = flag_value(row, f"bc_{hz}Hz_nr", first_nr_flag(row, bc_cols))

        bc_missing = clean_value_zero(row.get(bc_flag_col, 1.0 if bc is None else 0.0)) >= 0.5
        abg = row_numeric(row, f"abg_{hz}Hz")
        if abg is None and ac is not None and bc is not None:
            abg = ac - bc
        abg_missing = clean_value_zero(row.get(abg_flag_col, 1.0 if (ac is None or bc is None) else 0.0)) >= 0.5
        abg_censored = flag_value(row, abg_censored_col, 1.0 if ((ac_nr >= 0.5 or bc_nr >= 0.5) and not abg_missing) else 0.0)

        out[f"bc_{hz}Hz"] = 0.0 if bc is None else bc
        out[f"bc_{hz}Hz_nr"] = bc_nr
        out[bc_flag_col] = float(bc_missing)
        out[f"abg_{hz}Hz"] = 0.0 if abg is None else abg
        out[abg_flag_col] = float(abg_missing)
        out[abg_censored_col] = float(abg_censored)
        if not bc_missing and bc is not None:
            bc_values.append(bc)
        if not abg_missing and abg is not None:
            abg_values.append(abg)
            abg_censored_values.append(abg_censored)

    bc_mean = row_numeric(row, "bc_mean")
    abg_mean = row_numeric(row, "abg_mean")
    out["bc_mean"] = 0.0 if bc_mean is None else bc_mean
    if bc_mean is None:
        out["bc_mean"] = 0.0 if mean_present(bc_values) is None else mean_present(bc_values)
    out["abg_mean"] = 0.0 if abg_mean is None else abg_mean
    if abg_mean is None:
        out["abg_mean"] = 0.0 if mean_present(abg_values) is None else mean_present(abg_values)
    out["abg_mean_censored"] = flag_value(
        row,
        "abg_mean_censored",
        max(abg_censored_values) if abg_censored_values else 0.0,
    )
    return out


def build_task3_model_row(row: Mapping, side: str) -> dict:
    out = dict(row)
    side = normalize_side(side)
    out["ear_side"] = side
    for suffix, neutral_col in TASK3_NAME_MAP.items():
        legacy_col = f"tymp_{side}_{suffix}"
        raw_value = row.get(neutral_col, row.get(legacy_col, ""))
        real_zero, missing_zero, np_zero = task3_flags(raw_value)
        value = parse_numeric_value(raw_value)
        if np_zero >= 0.5 and neutral_col in TASK3_NP_EXTREME_VALUES:
            value = TASK3_NP_EXTREME_VALUES[neutral_col]
        out[neutral_col] = 0.0 if value is None else value
        for flag, default in [
            ("real_zero", real_zero),
            ("missing_zero", missing_zero),
            ("np_zero", np_zero),
        ]:
            neutral_flag = f"{neutral_col}_{flag}"
            legacy_flag = f"{legacy_col}_{flag}"
            out[neutral_flag] = clean_value_zero(row.get(neutral_flag, row.get(legacy_flag, default)))
    return out


def build_model_row(row: Mapping, task_name: str, side: str = "right") -> dict:
    side = normalize_side(row.get("ear_side", side))
    if task_name == "Task1":
        return build_task1_model_row(row, side)
    if task_name == "Task2":
        return build_task2_model_row(row, side)
    if task_name == "Task3":
        return build_task3_model_row(row, side)
    return dict(row)


def task1_degree_from_pta(pta):
    pta = clean_value_none(pta)
    if pta is None:
        return None
    if pta < 20:
        return "Normal hearing"
    if pta < 35:
        return "Mild hearing loss"
    if pta < 50:
        return "Moderate hearing loss"
    if pta < 65:
        return "Moderately severe hearing loss"
    if pta < 80:
        return "Severe hearing loss"
    if pta < 95:
        return "Profound hearing loss"
    return "Complete or total hearing loss"


def task1_evidence_status(row: Mapping) -> dict:
    has_pta = clean_value_none(row.get("ac_PTA")) is not None
    return {
        "evidence_status": "complete_evidence" if has_pta else "no_pta_data",
        "baseline_covered": bool(has_pta),
        "has_missing": bool(not has_pta),
        "rule_confidence": 1.0 if has_pta else 0.0,
        "warning_reasons": "" if has_pta else "no_pta_data",
    }


def task2_evidence_status(row: Mapping) -> dict:
    has_ac_missing = False
    has_bc_missing = False
    has_any_nr = False
    has_ac_nr = False
    has_bc_nr = False
    has_abg_censored = False
    has_abg_missing = False
    has_abg_borderline = False
    bc_present_count = 0
    ac_present_count = 0

    for hz in TASK2_ABG_FREQS:
        ac_col = f"ac_{hz}Hz"
        bc_col = f"bc_{hz}Hz"
        abg_col = f"abg_{hz}Hz"
        ac_value = clean_value_none(row.get(ac_col))
        bc_value = clean_value_none(row.get(bc_col))
        ac_nr = clean_value_zero(row.get(f"{ac_col}_nr")) >= 0.5
        bc_nr = clean_value_zero(row.get(f"{bc_col}_nr")) >= 0.5
        bc_missing_flag = clean_value_zero(row.get(f"{bc_col}_missing")) >= 0.5
        abg_missing_flag = clean_value_zero(row.get(f"{abg_col}_missing")) >= 0.5
        abg_censored_flag = clean_value_zero(row.get(f"{abg_col}_censored")) >= 0.5

        ac_missing = ac_value is None and not ac_nr
        bc_missing = (bc_value is None and not bc_nr) or bc_missing_flag
        abg_missing = abg_missing_flag or ac_missing or bc_missing
        abg_censored = abg_censored_flag or ac_nr or bc_nr

        has_ac_missing = has_ac_missing or ac_missing
        has_bc_missing = has_bc_missing or bc_missing
        has_ac_nr = has_ac_nr or ac_nr
        has_bc_nr = has_bc_nr or bc_nr
        has_any_nr = has_any_nr or ac_nr or bc_nr
        has_abg_missing = has_abg_missing or abg_missing
        has_abg_censored = has_abg_censored or abg_censored
        ac_present_count += int(ac_value is not None)
        bc_present_count += int(bc_value is not None)
        if not abg_missing and ac_value is not None and bc_value is not None:
            abg = ac_value - bc_value
            has_abg_borderline = has_abg_borderline or (8.0 <= abg <= 12.0)

    no_bc_data = bc_present_count == 0
    if no_bc_data:
        status = "no_bc_data"
    elif has_any_nr or has_abg_censored:
        status = "nr_or_censored"
    elif has_bc_missing:
        status = "bc_missing"
    elif has_ac_missing:
        status = "ac_missing"
    elif has_abg_missing:
        status = "abg_missing"
    elif has_abg_borderline:
        status = "abg_borderline"
    else:
        status = "complete_evidence"

    covered = status in {"complete_evidence", "abg_borderline"}
    warnings = []
    if no_bc_data:
        warnings.append("no_bc_data")
    if has_ac_missing:
        warnings.append("ac_missing")
    if has_bc_missing:
        warnings.append("bc_missing")
    if has_ac_nr:
        warnings.append("ac_nr")
    if has_bc_nr:
        warnings.append("bc_nr")
    if has_abg_missing:
        warnings.append("abg_missing")
    if has_abg_censored:
        warnings.append("abg_censored")
    if has_abg_borderline:
        warnings.append("abg_borderline")

    if status == "complete_evidence":
        confidence = 1.0
    elif status == "abg_borderline":
        confidence = 0.7
    elif status in {"bc_missing", "ac_missing", "abg_missing", "nr_or_censored"}:
        confidence = 0.5
    else:
        confidence = 0.25

    return {
        "evidence_status": status,
        "baseline_covered": bool(covered),
        "has_ac_missing": bool(has_ac_missing),
        "has_bc_missing": bool(has_bc_missing),
        "has_any_nr": bool(has_any_nr),
        "has_ac_nr": bool(has_ac_nr),
        "has_bc_nr": bool(has_bc_nr),
        "has_abg_missing": bool(has_abg_missing),
        "has_abg_censored": bool(has_abg_censored),
        "has_abg_borderline": bool(has_abg_borderline),
        "no_bc_data": bool(no_bc_data),
        "ac_present_count": int(ac_present_count),
        "bc_present_count": int(bc_present_count),
        "rule_confidence": float(confidence),
        "warning_reasons": ";".join(warnings),
    }


def task3_evidence_status(row: Mapping) -> dict:
    np_cols = [col for col in row if str(col).endswith("_np_zero")]
    missing_cols = [col for col in row if str(col).endswith("_missing_zero")]
    has_np = any(clean_value_zero(row.get(col)) >= 0.5 for col in np_cols)
    has_missing = any(clean_value_zero(row.get(col)) >= 0.5 for col in missing_cols)
    if has_missing:
        status = "missing_tymp_data"
    elif has_np:
        status = "np_evidence"
    else:
        status = "complete_evidence"
    return {
        "evidence_status": status,
        "baseline_covered": bool(not has_missing),
        "has_np": bool(has_np),
        "has_missing": bool(has_missing),
        "rule_confidence": 0.5 if has_missing else 1.0,
        "warning_reasons": "missing_tymp_data" if has_missing else ("np_evidence" if has_np else ""),
    }


def evidence_status(task_name: str, row: Mapping) -> dict:
    decision = shared_rule_decision(task_name, row)
    out = {
        "evidence_status": decision.evidence_status,
        "baseline_covered": bool(decision.covered),
        "rule_confidence": float(decision.confidence),
        "warning_reasons": ";".join(decision.warning_flags),
        "compatible_labels": "|".join(decision.compatible_labels),
    }
    if task_name == "Task1":
        out["has_missing"] = bool(not decision.covered)
        return out
    if task_name == "Task2":
        detail = task2_evidence_decision(row)
        for key, value in detail.items():
            if key in {"covered", "confidence", "warning_flags"}:
                continue
            out[key] = value
        return out
    if task_name == "Task3":
        out["has_np"] = bool(decision.evidence_status == "np_evidence")
        out["has_missing"] = bool(decision.evidence_status == "missing_tymp_data")
        return out
    raise KeyError(f"Unknown task: {task_name}")


def rule_prediction(task_name: str, row: Mapping):
    return shared_rule_decision(task_name, row).label


def normalize_prediction(value) -> str:
    if value is None:
        return INSUFFICIENT_EVIDENCE_LABEL
    text = str(value).strip()
    return text if text else INSUFFICIENT_EVIDENCE_LABEL


def clinical_warning_summary(task_name: str, row: Mapping, model_pred: Optional[str] = None) -> dict:
    decision = shared_rule_decision(task_name, row)
    evidence = evidence_status(task_name, row)
    forced_rule = normalize_prediction(decision.label)
    baseline_covered = bool(decision.covered)
    abstain_rule = forced_rule if baseline_covered else INSUFFICIENT_EVIDENCE_LABEL
    warning_reasons = list(decision.warning_flags)
    if not baseline_covered:
        warning_reasons.append("insufficient_rule_evidence")
    conflict = False
    if model_pred is not None and baseline_covered and forced_rule != INSUFFICIENT_EVIDENCE_LABEL:
        compatible = {str(item).strip() for item in decision.compatible_labels}
        compatible.add(str(decision.label).strip())
        conflict = str(model_pred).strip() not in compatible
        if conflict:
            warning_reasons.append("rule_model_conflict")
    return {
        **evidence,
        "forced_rule_label": forced_rule,
        "abstain_rule_label": abstain_rule,
        "compatible_labels": "|".join(decision.compatible_labels),
        "rule_model_conflict": bool(conflict),
        "warning_reasons": ";".join(dict.fromkeys(warning_reasons)),
    }
