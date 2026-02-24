"""Pytest configuration for tests."""

import sys
from pathlib import Path
import tempfile
import pandas as pd
import pytest

# Add parent directory (project root) to sys.path so that imports work
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session", autouse=True)
def setup_features_csv():
    """Create a temporary features_train.csv for tests if it doesn't exist.
    
    This ensures tests can run in CI environments without the data files.
    """
    features_path = Path("data/processed/features_train.csv")
    
    # Skip if file already exists
    if features_path.exists():
        return
    
    # Create minimal feature set with required columns for tests
    features = [
        "CODE_GENDER",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "CNT_CHILDREN",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "REGION_POPULATION_RELATIVE",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH",
        "OWN_CAR_AGE",
        "FLAG_MOBIL",
        "FLAG_EMP_PHONE",
        "FLAG_WORK_PHONE",
        "FLAG_CONT_MOBILE",
        "FLAG_PHONE",
        "FLAG_EMAIL",
        "CNT_FAM_MEMBERS",
        "REGION_RATING_CLIENT",
        "REGION_RATING_CLIENT_W_CITY",
        "HOUR_APPR_PROCESS_START",
        "REG_REGION_NOT_LIVE_REGION",
        "REG_REGION_NOT_WORK_REGION",
        "LIVE_REGION_NOT_WORK_REGION",
        "PAYMENT_RATE",
        "INCOME_CREDIT_PERC",
        "INCOME_PER_PERSON",
        "ANNUITY_INCOME_PERC",
        "DAYS_EMPLOYED_PERC",
        "NAME_CONTRACT_TYPE_Cash_loans",
        "NAME_CONTRACT_TYPE_Revolving_loans",
    ]
    
    # Create directory if it doesn't exist
    features_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create minimal dataframe and save
    df = pd.DataFrame({col: [0.0] for col in features})
    df.insert(0, "SK_ID_CURR", [1])
    df.insert(1, "TARGET", [0])
    df.to_csv(features_path, index=False)
