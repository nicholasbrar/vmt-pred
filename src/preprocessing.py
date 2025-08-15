import pandas as pd 
import os

HH_RAW = "data/nhts_raw/hh.csv"
VEH_RAW = "data/nhts_raw/veh.csv"
PROCESSED_PATH = "data/processed/hh_processed.csv"


def preprocess():
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

    veh = pd.read_csv(VEH_RAW)
    veh["ANNMILES"] = pd.to_numeric(veh["ANNMILES"], errors='coerce').fillna(0)
    hh_vmt = veh.groupby("HOUSEID")["ANNMILES"].sum().reset_index().rename(columns={"ANNMILES":"VMT"})
    hh = hh.merge(hh_vmt, on="HOUSEID", how="left")
    hh["VMT"] = hh["VMT"].fillna(0)  

    hh["VEH_PER_ADULT"] = hh["HHVEHCNT"] / hh["NUMADLT"].replace(0,1)
    hh["INCOME_PER_VEHICLE"] = hh["HHFAMINC"] / hh["HHVEHCNT"].replace(0,1)

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    hh.to_csv(PROCESSED_PATH, index=False)
    print(f"Processed data saved to {PROCESSED_PATH}")

if __name__ == "__main__":
    preprocess()