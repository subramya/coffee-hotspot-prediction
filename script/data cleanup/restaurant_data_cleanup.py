import pandas as pd

INPUT_FILE = "../../data/DOHMH_New_York_City_Restaurant_Inspection_Results_20260325.csv"
OUTPUT_FILE = "../../data/manhattan_cafes.csv"


def main():
    df = pd.read_csv(INPUT_FILE, low_memory=False)

    df.columns = [col.strip() for col in df.columns]
    df = df[df["BORO"].astype(str).str.strip().str.upper() == "MANHATTAN"].copy()
    df = df.dropna(subset=["Latitude", "Longitude", "CAMIS", "DBA"])

    df["DBA"] = df["DBA"].astype(str).str.strip()
    df["CUISINE DESCRIPTION"] = df["CUISINE DESCRIPTION"].astype(str).str.strip()

    cuisine_keep = {
        "Coffee/Tea",
        "Bakery Products/Desserts",
        "Donuts",
        "Bagels/Pretzels",
    }

    name_pattern = (
        r"coffee|cafe|caf[eé]|espresso|starbucks|dunkin|"
        r"bakery|bagel|tea"
    )

    cuisine_match = df["CUISINE DESCRIPTION"].isin(cuisine_keep)
    name_match = df["DBA"].str.contains(name_pattern, case=False, na=False, regex=True)

    df = df[cuisine_match | name_match].copy()

    df = df.sort_values(["CAMIS", "DBA"]).drop_duplicates(subset=["CAMIS"])

    cleaned = df[
        [
            "CAMIS",
            "DBA",
            "BORO",
            "CUISINE DESCRIPTION",
            "Latitude",
            "Longitude",
            "ZIPCODE",
            "STREET",
            "BUILDING",
        ]
    ].copy()

    cleaned = cleaned.rename(
        columns={
            "DBA": "business_name",
            "BORO": "boro",
            "CUISINE DESCRIPTION": "cuisine_description",
            "Latitude": "latitude",
            "Longitude": "longitude",
            "ZIPCODE": "zipcode",
            "STREET": "street",
            "BUILDING": "building",
        }
    )

    cleaned.to_csv(OUTPUT_FILE, index=False)

    print("Saved:", OUTPUT_FILE)
    print("Rows:", len(cleaned))
    print(cleaned.head())


if __name__ == "__main__":
    main()