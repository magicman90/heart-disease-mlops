"""
Execution:
    python src/eda.py
"""

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns 

from preprocessing import ( 
    load_raw,
    clean_and_binarize,
    NUMERIC_FEATURES,
    BINARY_TARGET_COLUMN,
)

sns.set_theme(style="whitegrid")

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "heart_disease_uci_raw.csv")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")


def run_eda():
    os.makedirs(FIG_DIR, exist_ok=True)
    raw = load_raw(RAW_PATH)

    print("=== Missing value analysis (raw data) ===")
    missing = raw.isin(["?"]).sum()
    missing = missing[missing > 0]
    print(missing if len(missing) else "No '?' missing markers found in raw columns shown as strings.")

    df = clean_and_binarize(raw)

    print("\n=== Missing values after numeric coercion ===")
    print(df.isna().sum())

    # Missing values bar chart
    na_counts = df.isna().sum()
    plt.figure(figsize=(8, 5))
    na_counts.plot(kind="bar", color="#4C72B0")
    plt.title("Missing Values per Feature")
    plt.ylabel("Count of Missing Values")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "missing_values.png"), dpi=150)
    plt.close()

    # Class balance
    plt.figure(figsize=(5, 5))
    counts = df[BINARY_TARGET_COLUMN].value_counts().sort_index()
    labels = ["No Disease (0)", "Disease (1)"]
    plt.pie(counts, labels=labels, autopct="%1.1f%%", colors=["#55A868", "#C44E52"])
    plt.title("Class Balance: Heart Disease Presence")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "class_balance.png"), dpi=150)
    plt.close()

    # Histograms of numeric features
    df[NUMERIC_FEATURES].hist(figsize=(12, 8), bins=20, color="#4C72B0", edgecolor="black")
    plt.suptitle("Distribution of Numeric Features")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "histograms.png"), dpi=150)
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()

    # Feature relationship
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x="age", y="thalach", hue=BINARY_TARGET_COLUMN, palette=["#55A868", "#C44E52"])
    plt.title("Age vs Max Heart Rate by Diagnosis")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "age_vs_thalach.png"), dpi=150)
    plt.close()

    print(f"\nSaved 5 EDA figures to {FIG_DIR}")


if __name__ == "__main__":
    run_eda()
