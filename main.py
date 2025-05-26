import pandas as pd
import joblib
from pathlib import Path

THRESHOLD = 0.193
MARGIN = 0.03
HIGH_VALUE_PRICES = ["60000 to 69000", "more than 69000"]

binary_maps = {
    "AccidentArea": {"Rural": 0, "Urban": 1},
    "Sex": {"Female": 0, "Male": 1},
    "Fault": {"Third Party": 0, "Policy Holder": 1},
    "PoliceReportFiled": {"No": 0, "Yes": 1},
    "WitnessPresent": {"No": 0, "Yes": 1},
    "AgentType": {"External": 0, "Internal": 1},
}

vehicle_price_labels = {
    "less than 20000": 0,
    "20000 to 29000": 1,
    "30000 to 39000": 2,
    "40000 to 59000": 3,
    "60000 to 69000": 4,
    "more than 69000": 5,
}
age_vehicle_labels = {
    "new": 0,
    "2 years": 1,
    "3 years": 2,
    "4 years": 3,
    "5 years": 4,
    "6 years": 5,
    "7 years": 6,
    "more than 7": 7,
}
base_policy_labels = {"Liability": 0, "Collision": 1, "All Perils": 2}

onehot_cols = [
    "Make",
    "MonthClaimed",
    "MaritalStatus",
    "PolicyType",
    "VehicleCategory",
    "RepNumber",
    "Deductible",
    "Days_Policy_Accident",
    "Days_Policy_Claim",
    "PastNumberOfClaims",
    "AgeOfPolicyHolder",
    "NumberOfSuppliments",
    "AddressChange_Claim",
    "NumberOfCars",
    "Year",
]

useless_cols = [
    "Month",
    "WeekOfMonth",
    "DayOfWeek",
    "DayOfWeekClaimed",
    "WeekOfMonthClaimed",
    "PolicyNumber",
]

template_cols = (
    pd.read_parquet("artifacts/df_clean3.parquet")
    .drop(columns=["FraudFound_P"])
    .columns.tolist()
)

model = joblib.load("artifacts/fraud_xgb_pipeline.pkl")


def _preprocess(d: dict) -> pd.DataFrame:
    df = pd.DataFrame([d])
    for col, m in binary_maps.items():
        df[col] = df[col].map(m)
    df["VehiclePrice"] = df["VehiclePrice"].map(vehicle_price_labels)
    df["AgeOfVehicle"] = df["AgeOfVehicle"].map(age_vehicle_labels)
    df["BasePolicy"] = df["BasePolicy"].map(base_policy_labels)
    df = pd.get_dummies(df, columns=onehot_cols, dtype=int)
    df.drop(columns=useless_cols, errors="ignore", inplace=True)
    for c in template_cols:
        if c not in df:
            df[c] = 0
    df = df[template_cols]
    return df


def classify_claim(raw: dict) -> str:
    df = _preprocess(raw)
    proba = float(model.predict_proba(df)[:, 1][0])
    if (
        raw["VehiclePrice"] in HIGH_VALUE_PRICES
        or abs(proba - THRESHOLD) < MARGIN
    ):
        return "further analysis needed"
    return "Fraudulent claim" if proba >= THRESHOLD else "Valid claim"