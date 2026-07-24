import pandas as pd


def create_engineered_features(df):
    """
    Create engineered features exactly as used during model training.
    """

    # -----------------------------
    # Tenure Group
    # -----------------------------
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-1 Year", "1-2 Years", "2-4 Years", "4-6 Years"],
        include_lowest=True
    )

    # -----------------------------
    # Average Monthly Spend
    # -----------------------------
    df["AvgMonthlySpend"] = df.apply(
        lambda row: 0 if row["tenure"] == 0
        else row["TotalCharges"] / row["tenure"],
        axis=1
    )

    # -----------------------------
    # New Customer
    # -----------------------------
    df["IsNewCustomer"] = (df["tenure"] < 12).astype(int)

    # -----------------------------
    # Long Contract
    # -----------------------------
    df["LongTermContract"] = (
        df["Contract"] != "Month-to-month"
    ).astype(int)

    # -----------------------------
    # Total Services
    # -----------------------------
    services = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    df["TotalServices"] = (df[services] == "Yes").sum(axis=1)

    # -----------------------------
    # Security Bundle
    # -----------------------------
    df["SecurityBundle"] = (
        (df["OnlineSecurity"] == "Yes")
        &
        (df["TechSupport"] == "Yes")
    ).astype(int)

    # -----------------------------
    # Streaming User
    # -----------------------------
    df["StreamingUser"] = (
        (df["StreamingTV"] == "Yes")
        |
        (df["StreamingMovies"] == "Yes")
    ).astype(int)

    # -----------------------------
    # Auto Payment
    # -----------------------------
    df["AutoPayment"] = (
        df["PaymentMethod"].isin([
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ])
    ).astype(int)

    # -----------------------------
    # Premium Customer
    # -----------------------------
    median_charge = 70.35

    df["PremiumCustomer"] = (
        df["MonthlyCharges"] > median_charge
    ).astype(int)

    return df