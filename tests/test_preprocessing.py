import pandas as pd

from src.preprocessing import RawToModelTransformer


def test_transform_computes_derived_features():
    row = pd.DataFrame([
        {
            "AMT_ANNUITY": 1000.0,
            "AMT_CREDIT": 20000.0,
            "AMT_INCOME_TOTAL": 60000.0,
            "CNT_FAM_MEMBERS": 3,
            "DAYS_EMPLOYED": -1000,
            "DAYS_BIRTH": -10000,
            "NAME_CONTRACT_TYPE": "Cash loans",
        }
    ])

    pre = RawToModelTransformer()
    out = pre.transform(row)

    # Derived numeric
    assert "PAYMENT_RATE" in out.columns
    assert abs(out["PAYMENT_RATE"].iloc[0] - (1000.0 / 20000.0)) < 1e-8
    assert "INCOME_CREDIT_PERC" in out.columns
    assert abs(out["INCOME_CREDIT_PERC"].iloc[0] - (60000.0 / 20000.0)) < 1e-8
    assert "INCOME_PER_PERSON" in out.columns
    assert abs(out["INCOME_PER_PERSON"].iloc[0] - (60000.0 / 3.0)) < 1e-8
    assert "ANNUITY_INCOME_PERC" in out.columns
    assert abs(out["ANNUITY_INCOME_PERC"].iloc[0] - (1000.0 / 60000.0)) < 1e-8
    assert "DAYS_EMPLOYED_PERC" in out.columns
    assert abs(out["DAYS_EMPLOYED_PERC"].iloc[0] - (-1000.0 / -10000.0)) < 1e-8


def test_transform_maps_categorical_to_one_hot():
    row = pd.DataFrame([
        {"NAME_CONTRACT_TYPE": "Cash loans", "AMT_INCOME_TOTAL": 1000.0}
    ])
    pre = RawToModelTransformer()
    out = pre.transform(row)

    # Expect a one-hot column for the contract type (sanitized name)
    # We look for any column that starts with NAME_CONTRACT_TYPE_ and contains 'Cash'
    matching = [c for c in out.columns if c.startswith("NAME_CONTRACT_TYPE_") and "Cash" in c]
    assert matching, "No one-hot column found for NAME_CONTRACT_TYPE"
    # the matching column should be 1 for our input
    assert out[matching[0]].iloc[0] == 1
