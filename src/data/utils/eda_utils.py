import pandas as pd

def explore_dataframe(df, name="DATAFRAME", target_column=None):
    print("\n" + "=" * 70)
    print(f"DATA EXPLORATION: {name}")
    print("=" * 70)

    # 1. Shape
    print("\n1. Dataset Shape:")
    print(df.shape)

    # 2. Columns
    print("\n2. Column Names:")
    print(df.columns.tolist())

    # 3. Data types
    print("\n3. Data Types:")
    print(df.dtypes)

    # 4. Missing values
    print("\n4. Missing Values:")
    print(df.isnull().sum())

    # 5. Duplicate rows
    print("\n5. Duplicate Rows:")
    print(df.duplicated().sum())

    # 6. Descriptive statistics
    print("\n6. Descriptive Statistics:")
    print(df.describe())

    # 7. Target statistics
    if target_column and target_column in df.columns:
        print(f"\n7. Target Statistics: {target_column}")
        print(df[target_column].describe())

    # 8. Unique values
    print("\n8. Unique Values (selected columns):")

    categorical_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    binary_cols = [
        col for col in df.columns
        if df[col].nunique() <= 5
    ]

    selected_cols = list(set(categorical_cols + binary_cols))

    for col in selected_cols[:15]:
        print(f"\n{col}:")
        print(df[col].unique())

    # 9. Correlation with target
    if target_column and target_column in df.columns:
        print(f"\n9. Top Correlations with {target_column}:")

        corr = (
            df.corr(numeric_only=True)[target_column]
            .sort_values(ascending=False)
        )

        print(corr.head(15))

    # 10. Outlier inspection
    if target_column and target_column in df.columns:
        print(f"\n10. Highest {target_column} values:")
        print(df.nlargest(10, target_column)[
            ["datetime", target_column]
        ])

    # 11. Head
    print("\n11. First 5 Rows:")
    print(df.head())

    # 12. Tail
    print("\n12. Last 5 Rows:")
    print(df.tail())