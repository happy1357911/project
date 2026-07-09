import math
import re
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


TASK2_ABG_FREQS = [500, 1000, 2000, 4000]
TASK2_RULE_CORE_AC_FREQS = [500, 1000, 2000, 4000]
TASK2_RULE_HIGH_AC_EITHER_FREQS = [6000, 8000]
TASK2_RULE_AC_FREQS = [*TASK2_RULE_CORE_AC_FREQS, *TASK2_RULE_HIGH_AC_EITHER_FREQS]
TASK2_EAR_LEVEL_AC_FREQS = [250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000]
TASK2_STANDARD_HEARING_TYPES = {"WNL", "SNHL", "CHL", "MHL"}
TASK2_RULE_FIRST_ENABLED_LABELS = {"WNL", "SNHL"}

TASK2_AC_NR_LIMITS_DB = {
    250: 110.0,
    500: 120.0,
    750: 120.0,
    1000: 120.0,
    1500: 120.0,
    2000: 120.0,
    3000: 120.0,
    4000: 120.0,
    6000: 115.0,
    8000: 110.0,
}

TASK2_BC_NR_LIMITS_DB = {
    500: 70.0,
    1000: 75.0,
    2000: 75.0,
    4000: 75.0,
}

TASK2_RULE_ABG_THRESHOLD_DB = 15.0
TASK2_ABG_BORDERLINE_LOW_DB = 10.0
TASK2_ABG_BORDERLINE_HIGH_DB = 15.0
TASK2_NEGATIVE_ABG_WARNING_DB = 15.0
TASK2_RULE_AC_THRESHOLD_DB = 25.0
TASK2_RULE_BC_THRESHOLD_DB = 25.0
RULE_FIRST_SCORE_THRESHOLD = 0.8
TASK3_C_PEAK_LOWER_DAPA = -300.0
TASK3_C_PEAK_UPPER_DAPA = -150.0
TASK3_LOW_COMPLIANCE_MMHO = 0.20
TASK3_WIDE_WIDTH_DAPA = 200.0

TASK3_NAME_MAP = {
    "Vea": "tymp_Vea",
    "peak_daPa": "tymp_peak_daPa",
    "peak_mmho": "tymp_peak_mmho",
    "Width_daPa": "tymp_Width_daPa",
}

TASK3_NP_EXTREME_VALUES = {
    "tymp_peak_daPa": -999.0,
    "tymp_peak_mmho": -1.0,
}

TASK3_SIDE_NP_EXTREME_VALUES = {
    "tymp_right_peak_daPa": -999.0,
    "tymp_left_peak_daPa": -999.0,
    "tymp_right_peak_mmho": -1.0,
    "tymp_left_peak_mmho": -1.0,
}

MISSING_TOKENS = {"", "nan", "NaN", "NA", "na", "N/A", "n/a", "None", "none", "NULL", "null"}

HARD_RULE_WARNING_TOKENS = (
    "no_bc_data",
    "ac_missing",
    "bc_missing",
    "abg_missing",
    "abg_borderline",
    "negative_abg_or_measurement_inconsistency",
    "high_frequency_ac_abnormal_support_only",
    "task2_rule_combo_uncertain",
    "task2_class_rule_precision_gate",
    "missing_tymp_peak",
    "tymp_bc_uncertain",
    "tymp_a_b_uncertain",
    "insufficient_rule_evidence",
    "incomplete_rule_data",
    "scenario_forced_model_fallback",
)


def has_hard_rule_warning(warning_text) -> bool:
    text = str(warning_text or "")
    return any(token in text for token in HARD_RULE_WARNING_TOKENS)


@dataclass(frozen=True)
class RuleDecision:
    label: Optional[str]
    confidence: float
    covered: bool
    evidence_status: str
    compatible_labels: Tuple[str, ...] = ()
    warning_flags: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        out = asdict(self)
        out["compatible_labels"] = list(self.compatible_labels)
        out["warning_flags"] = list(self.warning_flags)
        return out


def sanitize_column_name(col):
    return str(col).strip()


def hz_from_col(col):
    match = re.search(r"(\d+)Hz", str(col))
    return int(match.group(1)) if match else None


def parse_numeric_value(value, hz=None, nr_limits: Optional[Dict[int, float]] = None):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    raw = str(value).strip()
    if raw in MISSING_TOKENS or raw.lower() in MISSING_TOKENS:
        return None

    upper = raw.upper().replace(" ", "")
    is_nr = upper.endswith("NR")
    cleaned = upper[:-2] if is_nr else upper
    cleaned = cleaned.replace("DB", "")

    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None

    try:
        numeric = float(match.group(0))
    except ValueError:
        return None

    if not math.isfinite(numeric):
        return None
    if is_nr and hz is not None and nr_limits is not None:
        return float(nr_limits.get(hz, numeric))
    return numeric


def clean_value_none(value):
    return parse_numeric_value(value)


def clean_value_zero(value, hz=None, nr_limits: Optional[Dict[int, float]] = None):
    parsed = parse_numeric_value(value, hz=hz, nr_limits=nr_limits)
    return 0.0 if parsed is None else parsed


def is_nr_value(value) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass
    upper = str(value).strip().upper().replace(" ", "")
    upper = upper.replace("DB", "")
    return re.fullmatch(r"-?\d+(?:\.\d+)?NR", upper) is not None

def numeric_series(df: pd.DataFrame, col: str, nr_limits: Optional[Dict[int, float]] = None) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    hz = hz_from_col(col)
    parsed = df[col].map(lambda value: parse_numeric_value(value, hz=hz, nr_limits=nr_limits))
    return pd.to_numeric(parsed, errors="coerce")


def nr_indicator_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    return df[col].map(lambda value: 1.0 if is_nr_value(value) else 0.0).astype(float)


def flag_value(row: Mapping, col: str, default: float = 0.0) -> bool:
    value = parse_numeric_value(row.get(col, default))
    return value is not None and value >= 0.5


def mean_present(values: Iterable):
    cleaned = [parse_numeric_value(value) for value in values]
    cleaned = [value for value in cleaned if value is not None]
    if not cleaned:
        return None
    return float(np.mean(cleaned))


def task1_degree_from_pta(pta):
    pta = parse_numeric_value(pta)
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


def row_numeric_value(row: Mapping, col: str, nr_limits: Optional[Dict[int, float]] = None):
    return parse_numeric_value(row.get(col), hz=hz_from_col(col), nr_limits=nr_limits)


def infer_task2_hearing_type_from_rule(row: Mapping) -> Tuple[Optional[str], bool]:
    """Return (rule_label, has_uncertain_evidence) for Task2 ear-level rows.

    Task2 ABG is directional. Only AC worse than BC is clinically meaningful
    for the rule path; BC worse than AC is treated as measurement inconsistency.
    """
    any_core_ac_high = False
    any_bc_high = False
    has_clear_abg = False
    has_borderline_abg = False
    has_negative_abg = False
    high_ac_abnormal = False
    has_uncertain_evidence = False

    for hz in TASK2_RULE_CORE_AC_FREQS:
        ac_col = f"ac_{hz}Hz"
        ac = row_numeric_value(row, ac_col, TASK2_AC_NR_LIMITS_DB)
        ac_nr = flag_value(row, f"{ac_col}_nr")
        ac_missing = ac is None and not ac_nr
        has_uncertain_evidence = has_uncertain_evidence or ac_missing
        if ac is not None and ac > TASK2_RULE_AC_THRESHOLD_DB:
            any_core_ac_high = True

    for hz in TASK2_RULE_HIGH_AC_EITHER_FREQS:
        ac_col = f"ac_{hz}Hz"
        ac = row_numeric_value(row, ac_col, TASK2_AC_NR_LIMITS_DB)
        if ac is not None and ac > TASK2_RULE_AC_THRESHOLD_DB:
            high_ac_abnormal = True

    for hz in TASK2_ABG_FREQS:
        ac_col = f"ac_{hz}Hz"
        bc_col = f"bc_{hz}Hz"
        abg_col = f"abg_{hz}Hz"

        ac = row_numeric_value(row, ac_col, TASK2_AC_NR_LIMITS_DB)
        bc = row_numeric_value(row, bc_col, TASK2_BC_NR_LIMITS_DB)
        ac_nr = flag_value(row, f"{ac_col}_nr")
        bc_nr = flag_value(row, f"{bc_col}_nr")
        ac_missing = ac is None and not ac_nr
        bc_missing = flag_value(row, f"{bc_col}_missing") or (bc is None and not bc_nr)
        abg_missing = (
            flag_value(row, f"{abg_col}_missing", default=1.0 if ac_missing or bc_missing else 0.0)
            or ac_missing
            or bc_missing
        )

        has_uncertain_evidence = has_uncertain_evidence or ac_missing or bc_missing or abg_missing

        if (not bc_missing) and bc is not None and bc > TASK2_RULE_BC_THRESHOLD_DB:
            any_bc_high = True

        if (not abg_missing) and ac is not None and bc is not None:
            directional_abg = ac - bc
            has_clear_abg = has_clear_abg or (directional_abg >= TASK2_RULE_ABG_THRESHOLD_DB)
            has_borderline_abg = has_borderline_abg or (
                TASK2_ABG_BORDERLINE_LOW_DB <= directional_abg < TASK2_RULE_ABG_THRESHOLD_DB
            )
            has_negative_abg = has_negative_abg or (directional_abg <= -TASK2_NEGATIVE_ABG_WARNING_DB)

    if has_uncertain_evidence or has_negative_abg or has_borderline_abg:
        return None, True
    if high_ac_abnormal and not any_core_ac_high:
        return None, True

    if has_clear_abg and any_core_ac_high and any_bc_high:
        return "MHL", False
    if has_clear_abg and (not any_core_ac_high) and (not any_bc_high):
        return "CHL", False
    if (not has_clear_abg) and any_core_ac_high and any_bc_high:
        return "SNHL", False
    if (not has_clear_abg) and (not any_core_ac_high) and (not any_bc_high):
        return "WNL", False
    return None, True

def task2_evidence_decision(row: Mapping) -> dict:
    has_ac_missing = False
    has_core_ac_missing = False
    has_high_ac_pair_missing = False
    has_high_ac_abnormal = False
    has_bc_missing = False
    has_any_nr = False
    has_ac_nr = False
    has_bc_nr = False
    has_abg_censored = False
    has_abg_missing = False
    has_abg_borderline = False
    has_negative_abg = False
    has_clear_abg = False
    bc_present_count = 0
    bc_missing_count = 0
    ac_present_count = 0
    high_ac_present_count = 0

    for hz in TASK2_RULE_AC_FREQS:
        ac_col = f"ac_{hz}Hz"
        ac_value = parse_numeric_value(row.get(ac_col), hz=hz, nr_limits=TASK2_AC_NR_LIMITS_DB)
        ac_nr = flag_value(row, f"{ac_col}_nr")
        ac_missing = ac_value is None and not ac_nr
        has_ac_missing = has_ac_missing or ac_missing
        if hz in TASK2_RULE_CORE_AC_FREQS:
            has_core_ac_missing = has_core_ac_missing or ac_missing
        if hz in TASK2_RULE_HIGH_AC_EITHER_FREQS and not ac_missing and ac_value is not None:
            high_ac_present_count += 1
            has_high_ac_abnormal = has_high_ac_abnormal or (ac_value > TASK2_RULE_AC_THRESHOLD_DB)
        has_ac_nr = has_ac_nr or ac_nr
        has_any_nr = has_any_nr or ac_nr
        ac_present_count += int(ac_value is not None)

    has_high_ac_pair_missing = high_ac_present_count == 0

    for hz in TASK2_ABG_FREQS:
        ac_col = f"ac_{hz}Hz"
        bc_col = f"bc_{hz}Hz"
        abg_col = f"abg_{hz}Hz"
        ac_value = parse_numeric_value(row.get(ac_col), hz=hz, nr_limits=TASK2_AC_NR_LIMITS_DB)
        bc_value = parse_numeric_value(row.get(bc_col), hz=hz, nr_limits=TASK2_BC_NR_LIMITS_DB)
        ac_nr = flag_value(row, f"{ac_col}_nr")
        bc_nr = flag_value(row, f"{bc_col}_nr")
        bc_missing_flag = flag_value(row, f"{bc_col}_missing")
        abg_missing_flag = flag_value(row, f"{abg_col}_missing")
        abg_censored_flag = flag_value(row, f"{abg_col}_censored")

        ac_missing = ac_value is None and not ac_nr
        bc_missing = (bc_value is None and not bc_nr) or bc_missing_flag
        abg_missing = abg_missing_flag or ac_missing or bc_missing
        abg_censored = abg_censored_flag or ac_nr or bc_nr

        has_bc_missing = has_bc_missing or bc_missing
        bc_missing_count += int(bool(bc_missing))
        has_bc_nr = has_bc_nr or bc_nr
        has_any_nr = has_any_nr or ac_nr or bc_nr
        has_abg_missing = has_abg_missing or abg_missing
        has_abg_censored = has_abg_censored or abg_censored
        bc_present_count += int(bc_value is not None)
        if not abg_missing and ac_value is not None and bc_value is not None:
            directional_abg = ac_value - bc_value
            has_clear_abg = has_clear_abg or (directional_abg >= TASK2_RULE_ABG_THRESHOLD_DB)
            has_abg_borderline = has_abg_borderline or (
                TASK2_ABG_BORDERLINE_LOW_DB <= directional_abg < TASK2_RULE_ABG_THRESHOLD_DB
            )
            has_negative_abg = has_negative_abg or (
                directional_abg <= -TASK2_NEGATIVE_ABG_WARNING_DB
            )

    rule_label, label_uncertain = infer_task2_hearing_type_from_rule(row)
    class_rule_gate_blocked = bool(
        rule_label is not None and rule_label not in TASK2_RULE_FIRST_ENABLED_LABELS
    )
    no_bc_data = bc_present_count == 0
    high_ac_only_abnormal = bool(has_high_ac_abnormal and rule_label is None)
    complete_for_rule = bool(
        not has_core_ac_missing
        and not has_bc_missing
        and not has_abg_missing
        and not no_bc_data
    )
    hard_warnings = []
    if no_bc_data:
        hard_warnings.append("no_bc_data")
    if has_core_ac_missing:
        hard_warnings.append("ac_missing")
    if has_bc_missing:
        hard_warnings.append("bc_missing")
    if has_abg_missing:
        hard_warnings.append("abg_missing")
    if has_negative_abg:
        hard_warnings.append("negative_abg_or_measurement_inconsistency")
    if has_abg_borderline:
        hard_warnings.append("abg_borderline")
    if high_ac_only_abnormal:
        hard_warnings.append("high_frequency_ac_abnormal_support_only")
    if label_uncertain or rule_label is None:
        hard_warnings.append("task2_rule_combo_uncertain")
    if class_rule_gate_blocked:
        hard_warnings.append("task2_class_rule_precision_gate")

    if no_bc_data:
        status = "no_bc_data"
    elif has_negative_abg:
        status = "negative_abg_or_measurement_inconsistency"
    elif has_bc_missing:
        status = "bc_missing"
    elif has_core_ac_missing:
        status = "ac_missing"
    elif has_abg_missing:
        status = "abg_missing"
    elif has_abg_borderline:
        status = "abg_borderline"
    elif high_ac_only_abnormal:
        status = "high_frequency_ac_abnormal_support_only"
    elif rule_label is None:
        status = "task2_rule_combo_uncertain"
    elif class_rule_gate_blocked:
        status = "task2_class_rule_precision_gate"
    else:
        status = "complete_evidence"

    warnings = []
    if no_bc_data:
        warnings.append("no_bc_data")
    if has_core_ac_missing:
        warnings.append("ac_missing")
    if has_high_ac_pair_missing:
        warnings.append("high_ac_missing")
    if has_high_ac_abnormal:
        warnings.append("high_frequency_ac_abnormal")
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
    if has_negative_abg:
        warnings.append("negative_abg_or_measurement_inconsistency")
    if high_ac_only_abnormal:
        warnings.append("high_frequency_ac_abnormal_support_only")
    if label_uncertain or rule_label is None:
        warnings.append("task2_rule_combo_uncertain")
    if class_rule_gate_blocked:
        warnings.append("task2_class_rule_precision_gate")

    score = 1.0
    score_deductions = []
    if has_core_ac_missing:
        score -= 0.35
        score_deductions.append("core_ac_missing:-0.35")
    if has_bc_missing:
        score -= 0.30
        score_deductions.append("bc_group_missing:-0.30")
        if bc_missing_count >= 2:
            score = min(score, 0.60)
            score_deductions.append("bc_missing_ge2:cap_0.60")
    if has_abg_missing and not (has_core_ac_missing or has_bc_missing or no_bc_data):
        score -= 0.20
        score_deductions.append("abg_missing:-0.20")
    if has_abg_censored:
        score -= 0.05
        score_deductions.append("abg_censored:-0.05")
    if has_abg_borderline:
        score = min(score - 0.10, 0.70)
        score_deductions.append("abg_borderline:-0.10_cap_0.70")
    if has_negative_abg:
        score = min(score, 0.50)
        score_deductions.append("negative_abg:cap_0.50")
    if high_ac_only_abnormal:
        score = min(score, 0.70)
        score_deductions.append("high_frequency_ac_only:cap_0.70")
    if label_uncertain or rule_label is None:
        score = min(score, 0.50)
        score_deductions.append("task2_rule_combo_uncertain:cap_0.50")
    if class_rule_gate_blocked:
        score = min(score, 0.70)
        score_deductions.append("task2_class_rule_precision_gate:cap_0.70")
    if no_bc_data:
        score = min(score, 0.50)
        score_deductions.append("no_bc_data:cap_0.50")

    confidence = max(0.0, min(1.0, float(score)))
    covered = bool(
        confidence >= RULE_FIRST_SCORE_THRESHOLD
        and complete_for_rule
        and rule_label is not None
        and not hard_warnings
    )
    return {
        "evidence_status": status,
        "covered": bool(covered),
        "complete_for_rule": bool(complete_for_rule),
        "confidence": float(confidence),
        "rule_evidence_score": float(confidence),
        "score_deductions": ";".join(score_deductions),
        "warning_flags": tuple(dict.fromkeys(warnings)),
        "hard_warning_flags": tuple(dict.fromkeys(hard_warnings)),
        "has_ac_missing": bool(has_core_ac_missing),
        "has_any_ac_missing": bool(has_ac_missing),
        "has_core_ac_missing": bool(has_core_ac_missing),
        "has_high_ac_pair_missing": bool(has_high_ac_pair_missing),
        "has_high_ac_abnormal": bool(has_high_ac_abnormal),
        "has_bc_missing": bool(has_bc_missing),
        "has_any_nr": bool(has_any_nr),
        "has_ac_nr": bool(has_ac_nr),
        "has_bc_nr": bool(has_bc_nr),
        "has_abg_missing": bool(has_abg_missing),
        "has_abg_censored": bool(has_abg_censored),
        "has_abg_borderline": bool(has_abg_borderline),
        "has_negative_abg": bool(has_negative_abg),
        "has_clear_abg": bool(has_clear_abg),
        "borderline_only": bool(has_abg_borderline and not has_clear_abg),
        "high_ac_only_abnormal": bool(high_ac_only_abnormal),
        "task2_rule_combo_uncertain": bool(label_uncertain or rule_label is None),
        "task2_class_rule_precision_gate": bool(class_rule_gate_blocked),
        "no_bc_data": bool(no_bc_data),
        "ac_present_count": int(ac_present_count),
        "bc_present_count": int(bc_present_count),
        "bc_missing_count": int(bc_missing_count),
        "high_ac_present_count": int(high_ac_present_count),
    }

def predict_task3_type_from_rule(row: Mapping) -> Optional[str]:
    peak_np = flag_value(row, "tymp_peak_daPa_np_zero")
    cmpl_np = flag_value(row, "tymp_peak_mmho_np_zero")
    peak_missing = flag_value(row, "tymp_peak_daPa_missing_zero")
    peak = None if peak_np or peak_missing else parse_numeric_value(row.get("tymp_peak_daPa"))
    cmpl = None if cmpl_np else parse_numeric_value(row.get("tymp_peak_mmho"))
    width = parse_numeric_value(row.get("tymp_Width_daPa"))

    if peak_np and cmpl_np:
        return "B"
    if peak is None:
        return None
    if peak <= TASK3_C_PEAK_UPPER_DAPA:
        return None
    if cmpl is not None and cmpl < TASK3_LOW_COMPLIANCE_MMHO:
        return None
    if width is not None and width >= TASK3_WIDE_WIDTH_DAPA:
        return None
    return "A"

def rule_decision(task_name: str, row: Mapping) -> RuleDecision:
    if task_name == "Task1":
        label = task1_degree_from_pta(row.get("ac_PTA"))
        if label is None:
            return RuleDecision(
                label=None,
                confidence=0.0,
                covered=False,
                evidence_status="no_pta_data",
                warning_flags=("no_pta_data",),
            )
        return RuleDecision(
            label=label,
            confidence=1.0,
            covered=True,
            evidence_status="complete_evidence",
            compatible_labels=(label,),
        )

    if task_name == "Task2":
        label, _ = infer_task2_hearing_type_from_rule(row)
        evidence = task2_evidence_decision(row)
        return RuleDecision(
            label=label,
            confidence=float(evidence["confidence"]),
            covered=bool(evidence["covered"]),
            evidence_status=str(evidence["evidence_status"]),
            compatible_labels=(label,) if label is not None else (),
            warning_flags=tuple(evidence["warning_flags"]),
        )

    if task_name == "Task3":
        peak_np = flag_value(row, "tymp_peak_daPa_np_zero")
        cmpl_np = flag_value(row, "tymp_peak_mmho_np_zero")
        peak_missing = flag_value(row, "tymp_peak_daPa_missing_zero")
        cmpl_missing = flag_value(row, "tymp_peak_mmho_missing_zero")
        width_missing = flag_value(row, "tymp_Width_daPa_missing_zero")
        vea_missing = flag_value(row, "tymp_Vea_missing_zero") or parse_numeric_value(row.get("tymp_Vea")) is None
        peak = None if peak_np or peak_missing else parse_numeric_value(row.get("tymp_peak_daPa"))
        cmpl = None if cmpl_np or cmpl_missing else parse_numeric_value(row.get("tymp_peak_mmho"))
        width = None if width_missing else parse_numeric_value(row.get("tymp_Width_daPa"))

        secondary_warnings = []
        score = 1.0
        if cmpl_missing and not cmpl_np:
            score -= 0.10
            secondary_warnings.append("tymp_peak_mmho_missing")
        if width_missing:
            score -= 0.10
            secondary_warnings.append("tymp_width_missing")
        if vea_missing:
            score -= 0.05
            secondary_warnings.append("tymp_vea_missing")

        low_compliance = bool(cmpl is not None and cmpl < TASK3_LOW_COMPLIANCE_MMHO)
        wide_width = bool(width is not None and width >= TASK3_WIDE_WIDTH_DAPA)

        if peak_np and cmpl_np:
            confidence = max(0.0, min(1.0, float(score)))
            return RuleDecision(
                label="B",
                confidence=confidence,
                covered=confidence >= RULE_FIRST_SCORE_THRESHOLD,
                evidence_status="np_evidence",
                compatible_labels=("B",),
                warning_flags=tuple(dict.fromkeys(["np_evidence", *secondary_warnings])),
            )
        if peak is None:
            confidence = max(0.0, min(0.50, float(score)))
            return RuleDecision(
                label=None,
                confidence=confidence,
                covered=False,
                evidence_status="missing_tymp_data",
                compatible_labels=(),
                warning_flags=tuple(dict.fromkeys(["missing_tymp_peak", *secondary_warnings])),
            )
        if peak <= TASK3_C_PEAK_LOWER_DAPA:
            confidence = max(0.0, min(0.50, float(score)))
            return RuleDecision(
                label=None,
                confidence=confidence,
                covered=False,
                evidence_status="tymp_bc_uncertain_extreme_negative_peak",
                compatible_labels=("B", "C"),
                warning_flags=tuple(dict.fromkeys(["tymp_bc_uncertain_extreme_negative_peak", *secondary_warnings])),
            )
        if peak <= TASK3_C_PEAK_UPPER_DAPA:
            confidence = max(0.0, min(0.60, float(score - 0.20)))
            return RuleDecision(
                label=None,
                confidence=confidence,
                covered=False,
                evidence_status="tymp_bc_uncertain_negative_peak",
                compatible_labels=("B", "C"),
                warning_flags=tuple(dict.fromkeys(["tymp_bc_uncertain_negative_peak", *secondary_warnings])),
            )
        if low_compliance or wide_width:
            warnings = []
            if low_compliance:
                warnings.append("tymp_low_compliance_a_b_uncertain")
            if wide_width:
                warnings.append("tymp_wide_width_a_b_uncertain")
            confidence = max(0.0, min(0.70, float(score)))
            return RuleDecision(
                label=None,
                confidence=confidence,
                covered=False,
                evidence_status="tymp_a_b_uncertain",
                compatible_labels=("A", "B"),
                warning_flags=tuple(dict.fromkeys([*warnings, *secondary_warnings])),
            )
        confidence = max(0.0, min(1.0, float(score)))
        return RuleDecision(
            label="A",
            confidence=confidence,
            covered=confidence >= RULE_FIRST_SCORE_THRESHOLD,
            evidence_status="complete_evidence",
            compatible_labels=("A",),
            warning_flags=tuple(dict.fromkeys(secondary_warnings)),
        )

    raise KeyError(f"Unknown task: {task_name}")

def score_rule_decision(task_name: str, row: Mapping, pred_name: str) -> float:
    decision = rule_decision(task_name, row)
    if not decision.covered or decision.label is None:
        return 0.0
    p = str(pred_name).strip().upper()
    label = str(decision.label).strip().upper()
    compatible = {str(item).strip().upper() for item in decision.compatible_labels}
    if p == label:
        return 1.0 if decision.confidence >= 1.0 else float(decision.confidence)
    if p in compatible:
        return float(decision.confidence)
    return -0.2


def score_task3_expert_consistency(row: Mapping, pred_name: str) -> float:
    """Return the Task3 expert-consistency target used by train/dashboard."""
    return score_rule_decision("Task3", row, pred_name)


def legacy_score_task3_expert_consistency(row: Mapping, pred_name: str) -> float:
    p = str(pred_name).strip().upper()
    peak_np = flag_value(row, "tymp_peak_daPa_np_zero")
    cmpl_np = flag_value(row, "tymp_peak_mmho_np_zero")
    peak_missing = flag_value(row, "tymp_peak_daPa_missing_zero")
    cmpl_missing = flag_value(row, "tymp_peak_mmho_missing_zero")
    width_missing = flag_value(row, "tymp_Width_daPa_missing_zero")

    peak = None if peak_np or peak_missing else parse_numeric_value(row.get("tymp_peak_daPa"))

    if p == "A":
        if peak is not None and -100 <= peak <= 100:
            return 1.0
        if peak is not None and -150 < peak < -100:
            return 0.5
        return -0.2

    if p == "B":
        if peak_np and cmpl_np and width_missing:
            return 1.0
        if width_missing and not peak_np and not cmpl_np and not peak_missing and not cmpl_missing:
            return 0.5
        return -0.2

    if p == "C":
        if peak is not None and peak <= -150:
            return 1.0
        if peak is not None and peak <= -100:
            return 0.5
        return -0.2

    return 0.0
