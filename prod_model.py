import pandas as pd
import joblib
from pathlib import Path
from sklearn.utils import shuffle
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

RANDOM_STATE = 42

best_params = {
    "n_estimators": 1900,
    "max_depth": 8,
    "learning_rate": 0.01687779530956748,
    "subsample": 0.6874195354945065,
    "colsample_bytree": 0.8163310047334191,
    "gamma": 1.8217752519279815,
    "min_child_weight": 2.0940811807759867,
    "reg_alpha": 0.5144801232675891,
    "reg_lambda": 0.6615578926346247,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

df = pd.read_csv("./data/fraud_oracle.csv")

binary_maps = {
    "AccidentArea": {"Rural": 0, "Urban": 1},
    "Sex": {"Female": 0, "Male": 1},
    "Fault": {"Third Party": 0, "Policy Holder": 1},
    "PoliceReportFiled": {"No": 0, "Yes": 1},
    "WitnessPresent": {"No": 0, "Yes": 1},
    "AgentType": {"External": 0, "Internal": 1},
}

for col, mapping in binary_maps.items():
    df[col] = df[col].map(mapping)

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

df["VehiclePrice"] = df["VehiclePrice"].map(vehicle_price_labels)
df["AgeOfVehicle"] = df["AgeOfVehicle"].map(age_vehicle_labels)
df["BasePolicy"] = df["BasePolicy"].map(base_policy_labels)

df = df[(df["Age"] != 0) & (df["AgeOfPolicyHolder"] != 0)]

onehot_cols = [
    "Age",
    "Make",
    "MaritalStatus",
    "PolicyType",
    "VehicleCategory",
    "RepNumber",
    "Deductible",
    "Days_Policy_Accident",
    "Days_Policy_Claim",
    "PastNumberOfClaims",
    "AgeOfPolicyHolder",
    "NumberOfCars"
]
df = pd.get_dummies(df, columns=onehot_cols, dtype=int)

useless_cols = [
    "Month",
    "WeekOfMonth",
    "DayOfWeek",
    "DayOfWeekClaimed",
    "WeekOfMonthClaimed",
    "PolicyNumber",
    "MonthClaimed",
    "NumberOfSuppliments",
    "AddressChange_Claim",
    "Year"
]
df.drop(columns=useless_cols, inplace=True)

Path("artifacts").mkdir(exist_ok=True)
df.to_parquet("artifacts/df_clean3.parquet", index=False)

X = df.drop(columns=["FraudFound_P"])
y = df["FraudFound_P"]
X, y = shuffle(X, y, random_state=RANDOM_STATE)

pipe = Pipeline(
    [("smote", SMOTE(random_state=RANDOM_STATE)), ("clf", XGBClassifier(**best_params))]
)

pipe.fit(X, y)
joblib.dump(pipe, "artifacts/fraud_xgb_pipeline.pkl")