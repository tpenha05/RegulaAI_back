import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

binary_columns = [
    "AccidentArea",
    "Sex",
    "Fault",
    "PoliceReportFiled",
    "WitnessPresent",
    "AgentType",
]

onehot_columns = [
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

ordinal_columns = ["VehiclePrice", "AgeOfVehicle", "BasePolicy"]

ordinal_categories = [
    [
        "less than 20000",
        "20000 to 29000",
        "30000 to 39000",
        "40000 to 59000",
        "60000 to 69000",
        "more than 69000",
    ],
    [
        "new",
        "2 years",
        "3 years",
        "4 years",
        "5 years",
        "6 years",
        "7 years",
        "more than 7",
    ],
    ["Liability", "Collision", "All Perils"],
]

binary_categories = [
    ["Rural", "Urban"],
    ["Female", "Male"],
    ["Policy Holder", "Third Party"],
    ["No", "Yes"],
    ["No", "Yes"],
    ["Internal", "External"],
]

useless_columns = [
    "Month",
    "WeekOfMonth",
    "DayOfWeek",
    "DayOfWeekClaimed",
    "WeekOfMonthClaimed",
    "PolicyNumber",
]

class ZeroCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        mask = (X["Age"] != 0) & (X["AgeOfPolicyHolder"] != 0)
        return X.loc[mask]

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.drop(columns=self.cols)

encoder = ColumnTransformer(
    transformers=[
        (
            "binary",
            OrdinalEncoder(
                categories=binary_categories,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
            binary_columns,
        ),
        (
            "ordinal",
            OrdinalEncoder(
                categories=ordinal_categories,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
            ordinal_columns,
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            onehot_columns,
        ),
    ],
    remainder="passthrough",
)

preprocess_pipeline = Pipeline(
    steps=[
        ("zero_cleaner", ZeroCleaner()),
        ("drop_useless", ColumnDropper(useless_columns)),
        ("encoder", encoder),
    ]
)

def fit_pipeline(df: pd.DataFrame, target: str = "FraudFound_P") -> None:
    X = df.drop(columns=[target])
    preprocess_pipeline.fit(X)
    joblib.dump(preprocess_pipeline, "preprocess.pkl")

def transform_payload(payload) -> pd.DataFrame:
    pipe = joblib.load("preprocess.pkl")
    if isinstance(payload, dict):
        payload = pd.DataFrame([payload])
    elif isinstance(payload, pd.Series):
        payload = payload.to_frame().T
    return pd.DataFrame(pipe.transform(payload), columns=pipe.get_feature_names_out())