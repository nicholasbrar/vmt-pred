import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

DATA_PATH = "data/processed/hh_processed.csv"
df = pd.read_csv(DATA_PATH)

df = df[df['VMT'].notna()].copy()
df['VMT'] = df['VMT'].clip(lower=0)
vmt_cap = df['VMT'].quantile(0.99)
df['VMT'] = df['VMT'].clip(upper=vmt_cap)

df['HHVEHCNT'] = df['HHVEHCNT'].clip(upper=df['HHVEHCNT'].quantile(0.99))
df['HHSIZE'] = df['HHSIZE'].clip(upper=df['HHSIZE'].quantile(0.99))
df['NUMADLT'] = df['NUMADLT'].clip(upper=df['NUMADLT'].quantile(0.99))

categorical = ["URBAN", "HOMETYPE", "HOMEOWN"]
df = pd.get_dummies(df, columns=categorical, drop_first=False)  

strong_dummies = ["URBAN_4", "URBAN_2", "HOMETYPE_3", "HOMETYPE_2", "HOMEOWN_3"]
dummies_to_drop = [c for c in df.columns if c in df.columns and c not in strong_dummies and any(cat in c for cat in categorical)]
df = df.drop(columns=dummies_to_drop)

features = [
    "NUMADLT", "HHVEHCNT", "HHFAMINC", "VEH_PER_ADULT", 
    "INCOME_PER_VEHICLE", "DRVRCNT", "HHSIZE", "VEHYEAR",
    "VEHAGE", "VEHCOMMERCIAL", "DRIVER", "WORKER", "R_AGE"
]
features += [c for c in df.columns if any(cat in c for cat in categorical)]

X = df[features]
X['INCOME_per_VEHICLE_times_VEH_PER_ADULT'] = X['INCOME_PER_VEHICLE'] * X['VEH_PER_ADULT']

y = df['VMT']
weights = df["WTHHFIN"]

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, weights, test_size=0.2, random_state=42
)

linreg = LinearRegression()
linreg.fit(X_train, y_train, sample_weight=w_train)
y_pred = linreg.predict(X_test)

errors = y_test - y_pred
weighted_rmse = np.sqrt(np.sum(w_test * errors**2) / np.sum(w_test))
weighted_mae = np.sum(w_test * np.abs(errors)) / np.sum(w_test)
r2 = r2_score(y_test, y_pred, sample_weight=w_test)

print(f"Weighted RMSE: {weighted_rmse:.2f}")
print(f"Weighted MAE: {weighted_mae:.2f}")
print(f"Weighted R²: {r2:.2f}\n")

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train, sample_weight=w_train)

importances = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("=== Random Forest Feature Importances ===")
print(importances.head(15))
