import os
import pandas as pd

_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def extract_age_income():
    # Read the adult dataset
    adult_df = pd.read_csv(os.path.join(_DATA_DIR, "adult.csv"))

    # Extract first (age) and last (income>50K) columns
    age_income_df = adult_df[['age', 'income>50K']]

    # Rename the income column to just 'income'
    age_income_df = age_income_df.rename(columns={'income>50K': 'income'})

    # Save to new CSV file
    age_income_df.to_csv(os.path.join(_DATA_DIR, "age_income.csv"), index=False)
    print("Successfully extracted age and income columns to age_income.csv")
    print(f"Number of rows: {len(age_income_df)}")
    
if __name__ == "__main__":
    extract_age_income() 