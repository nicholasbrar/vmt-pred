import pandas as pd
import os

HH_RAW = "data/nhts_raw/hh.csv"
VEH_RAW = "data/nhts_raw/veh.csv"
PER_RAW = "data/nhts_raw/pers.csv"
PROCESSED_PATH = "data/processed/hh_processed.csv"

def preprocess():
    # Household data
    hh = pd.read_csv(HH_RAW)
    hh = hh.dropna(subset=["HOUSEID"])
    
    numeric_cols = hh.select_dtypes(include="number").columns
    hh[numeric_cols] = hh[numeric_cols].fillna(hh[numeric_cols].median())

    cat_cols = hh.select_dtypes(include="object").columns
    for col in cat_cols:
        hh[col] = hh[col].fillna(hh[col].mode()[0])
    
    features = [
        "HOUSEID", "NUMADLT", "HHVEHCNT", "HHFAMINC", "URBAN",
        "HOMETYPE", "HOMEOWN", "WTHHFIN", "DRVRCNT", "HHSIZE"
    ]
    hh = hh[features]
    hh["WTHHFIN"] = hh["WTHHFIN"].fillna(hh["WTHHFIN"].median())

    # Vehicle data
    veh = pd.read_csv(VEH_RAW)
    veh["ANNMILES"] = pd.to_numeric(veh["ANNMILES"], errors='coerce').fillna(0)
    veh["VEHYEAR"] = pd.to_numeric(veh["VEHYEAR"], errors='coerce').fillna(veh["VEHYEAR"].median())
    veh["VEHAGE"] = pd.to_numeric(veh.get("VEHAGE", pd.Series(0)), errors='coerce').fillna(veh["VEHAGE"].median())
    veh["VEHCOMMERCIAL"] = pd.to_numeric(veh.get("VEHCOMMERCIAL", pd.Series(0)), errors='coerce').fillna(0)

    veh_agg = veh.groupby("HOUSEID").agg({
        "ANNMILES": "sum",           
        "VEHYEAR": "mean",           
        "VEHAGE": "mean",            
        "VEHCOMMERCIAL": "sum"       
    }).reset_index().rename(columns={"ANNMILES": "VMT"})
    
    hh = hh.merge(veh_agg, on="HOUSEID", how="left")
    hh["VMT"] = hh["VMT"].fillna(0)
    hh["VEHYEAR"] = hh["VEHYEAR"].fillna(hh["VEHYEAR"].median())
    hh["VEHAGE"] = hh["VEHAGE"].fillna(hh["VEHAGE"].median())
    hh["VEHCOMMERCIAL"] = hh["VEHCOMMERCIAL"].fillna(0)

    # Person data
    per = pd.read_csv(PER_RAW)

    base_person_features = ["HOUSEID", "DRIVER", "WORKER", "R_AGE"]

    extra_features = ["GCDWORK", "PTUSED", "DELIV_FOOD"]

    per = per[base_person_features + extra_features]

    per_agg = per.groupby("HOUSEID").agg({
        "DRIVER": "sum",
        "WORKER": "sum",
        "R_AGE": "mean",
        "GCDWORK": "sum",        
        "PTUSED": "max",          
        "DELIV_FOOD": "max"       
    }).reset_index()

    hh = hh.merge(per_agg, on="HOUSEID", how="left")
    hh["DRIVER"] = hh["DRIVER"].fillna(0)
    hh["WORKER"] = hh["WORKER"].fillna(0)
    hh["R_AGE"] = hh["R_AGE"].fillna(hh["R_AGE"].median())
    hh["GCDWORK"] = hh["GCDWORK"].fillna(0)
    hh["PTUSED"] = hh["PTUSED"].fillna(0)
    hh["DELIV_FOOD"] = hh["DELIV_FOOD"].fillna(0)

    hh["VEH_PER_ADULT"] = hh["HHVEHCNT"] / hh["NUMADLT"].replace(0, 1)
    hh["INCOME_PER_VEHICLE"] = hh["HHFAMINC"] / hh["HHVEHCNT"].replace(0, 1)

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    hh.to_csv(PROCESSED_PATH, index=False)
    print(f"Processed data saved to {PROCESSED_PATH}")

if __name__ == "__main__":
    preprocess()
