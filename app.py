
# ============================================================
# NASSAU CANDY DISTRIBUTOR
# SHIPPING OPTIMIZATION RECOMMENDATION SYSTEM
# Complete Streamlit Dashboard
# ============================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION AND FILE PATHS
# ============================================================

st.set_page_config(
    page_title="Nassau Candy Shipping Optimization",
    page_icon="🍭",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
EXCEL_FILE = BASE_DIR / "Nassau Candy Distributor.xlsx"
MODEL_FILE = BASE_DIR / "best_profit_model.joblib"
MODEL_RESULTS_FILE = BASE_DIR / "ml_model_comparison.csv"

st.title("🍭 Nassau Candy Distributor")
st.subheader("Shipping Optimization Recommendation System")
st.caption(
    "Sales analytics, shipping analysis, product insights, "
    "machine learning and data-quality investigation."
)


# ============================================================
# 2. LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    if not EXCEL_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {EXCEL_FILE.name}. "
            "Keep the Excel file in the same folder as app.py."
        )

    data = pd.read_excel(EXCEL_FILE)

    required_columns = [
        "Row ID", "Order ID", "Order Date", "Ship Date",
        "Ship Mode", "Customer ID", "Country/Region", "City",
        "State/Province", "Postal Code", "Division", "Region",
        "Product ID", "Product Name", "Sales", "Units",
        "Gross Profit", "Cost"
    ]

    missing = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing required Excel columns: " + ", ".join(missing)
        )

    for column in ["Order Date", "Ship Date"]:
        data[column] = pd.to_datetime(
            data[column], errors="coerce"
        )

    for column in ["Sales", "Units", "Gross Profit", "Cost"]:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    data["Shipping_Lead_Time"] = (
        data["Ship Date"] - data["Order Date"]
    ).dt.days

    data["ID Year"] = (
        data["Order ID"]
        .astype("string")
        .str.extract(r"^[A-Z]{2}-(\d{4})-", expand=False)
    )

    return data


@st.cache_resource
def load_ml_model():
    if not MODEL_FILE.exists():
        return None
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_model_results():
    if not MODEL_RESULTS_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(MODEL_RESULTS_FILE)


try:
    df = load_data()
except Exception as error:
    st.error(f"Unable to load the dataset: {error}")
    st.stop()


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def money(value):
    if pd.isna(value):
        return "N/A"
    return f"${value:,.2f}"


def profit_margin(sales, profit):
    if pd.isna(sales) or sales == 0:
        return 0.0
    return float(profit / sales * 100)


def show_empty():
    st.warning(
        "No records match the selected filters. "
        "Change the filters in the sidebar."
    )


def shipping_summary(data):
    if data.empty:
        return pd.DataFrame()

    result = (
        data.groupby("Ship Mode", dropna=False)
        .agg(
            Order_Count=("Order ID", "nunique"),
            Total_Units=("Units", "sum"),
            Total_Sales=("Sales", "sum"),
            Total_Cost=("Cost", "sum"),
            Total_Gross_Profit=("Gross Profit", "sum")
        )
        .reset_index()
    )

    result["Profit_Margin_Percent"] = (
        result["Total_Gross_Profit"]
        / result["Total_Sales"].replace(0, np.nan)
        * 100
    ).fillna(0)

    return result


# ============================================================
# 4. SIDEBAR FILTERS AND NAVIGATION
# ============================================================

st.sidebar.title("Dashboard Filters")

regions = sorted(df["Region"].dropna().unique().tolist())
modes = sorted(df["Ship Mode"].dropna().unique().tolist())
product_names = sorted(
    df["Product Name"].dropna().unique().tolist()
)

selected_regions = st.sidebar.multiselect(
    "Select Region", regions, default=regions
)

selected_modes = st.sidebar.multiselect(
    "Select Shipping Mode", modes, default=modes
)

selected_products = st.sidebar.multiselect(
    "Select Products", product_names, default=product_names
)

filtered_df = df[
    df["Region"].isin(selected_regions)
    & df["Ship Mode"].isin(selected_modes)
    & df["Product Name"].isin(selected_products)
].copy()

st.sidebar.divider()

page = st.sidebar.radio(
    "Choose a Page",
    [
        "Executive Dashboard",
        "Shipping Analysis",
        "Regional Analysis",
        "Product Insights",
        "Product Shipping Analysis",
        "Recommendation Engine",
        "ML Predictions",
        "What-If Simulator",
        "Data Quality & Risk"
    ]
)

st.sidebar.divider()
st.sidebar.caption(
    "Nassau Candy Distributor | Shipping Optimization Project"
)


# ============================================================
# PAGE 1: EXECUTIVE DASHBOARD
# ============================================================

if page == "Executive Dashboard":
    st.header("📊 Executive Dashboard")

    if filtered_df.empty:
        show_empty()
        st.stop()

    sales = filtered_df["Sales"].sum()
    cost = filtered_df["Cost"].sum()
    profit = filtered_df["Gross Profit"].sum()
    units = filtered_df["Units"].sum()
    orders = filtered_df["Order ID"].nunique()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Sales", money(sales))
    c2.metric("Gross Profit", money(profit))
    c3.metric("Total Cost", money(cost))
    c4.metric("Units Sold", f"{units:,.0f}")
    c5.metric("Unique Orders", f"{orders:,}")

    st.metric("Gross Profit Margin", f"{profit_margin(sales, profit):.2f}%")

    left, right = st.columns(2)

    with left:
        region_sales = (
            filtered_df.groupby("Region", as_index=False)
            .agg(Sales=("Sales", "sum"))
        )
        fig = px.bar(
            region_sales, x="Region", y="Sales",
            color="Region", title="Sales by Region",
            text_auto=".2s"
        )
        st.plotly_chart(fig, width="stretch")

    with right:
        summary = shipping_summary(filtered_df)
        fig = px.bar(
            summary, x="Ship Mode", y="Total_Gross_Profit",
            color="Ship Mode", title="Gross Profit by Shipping Mode",
            text_auto=".2s"
        )
        st.plotly_chart(fig, width="stretch")

    monthly = filtered_df.dropna(subset=["Order Date"]).copy()
    monthly["Order Month"] = (
        monthly["Order Date"].dt.to_period("M").astype(str)
    )
    monthly_sales = (
        monthly.groupby("Order Month", as_index=False)
        .agg(Sales=("Sales", "sum"))
        .sort_values("Order Month")
    )

    fig = px.line(
        monthly_sales, x="Order Month", y="Sales",
        markers=True, title="Monthly Sales Trend"
    )
    st.plotly_chart(fig, width="stretch")

    st.info(
        "Historical sales and gross profit are descriptive measures. "
        "They do not establish actual freight-cost savings."
    )


# ============================================================
# PAGE 2: SHIPPING ANALYSIS
# ============================================================

elif page == "Shipping Analysis":
    st.header("🚚 Shipping Analysis")

    if filtered_df.empty:
        show_empty()
        st.stop()

    summary = shipping_summary(filtered_df)
    st.dataframe(summary.round(2), width="stretch", hide_index=True)

    left, right = st.columns(2)

    with left:
        fig = px.bar(
            summary, x="Ship Mode", y="Order_Count",
            color="Ship Mode", title="Unique Orders by Shipping Mode",
            text_auto=True
        )
        st.plotly_chart(fig, width="stretch")

    with right:
        fig = px.bar(
            summary, x="Ship Mode", y="Profit_Margin_Percent",
            color="Ship Mode", title="Gross Profit Margin by Mode",
            text_auto=".2f"
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("Shipping Duration Statistics")

    lead = filtered_df.dropna(subset=["Shipping_Lead_Time"])

    if not lead.empty:
        lead_summary = (
            lead.groupby("Ship Mode")["Shipping_Lead_Time"]
            .agg(
                Record_Count="count",
                Minimum_Days="min",
                Median_Days="median",
                Maximum_Days="max",
                Average_Days="mean"
            )
            .reset_index()
        )
        st.dataframe(
            lead_summary.round(2),
            width="stretch",
            hide_index=True
        )

        st.warning(
            "The supplied dates produce unusually long durations, "
            "including for Same Day shipments. Validate these dates "
            "before using shipping duration for prediction."
        )


# ============================================================
# PAGE 3: REGIONAL ANALYSIS
# ============================================================

elif page == "Regional Analysis":
    st.header("🌎 Regional Analysis")

    if filtered_df.empty:
        show_empty()
        st.stop()

    regional = (
        filtered_df.groupby("Region", as_index=False)
        .agg(
            Unique_Orders=("Order ID", "nunique"),
            Units=("Units", "sum"),
            Sales=("Sales", "sum"),
            Cost=("Cost", "sum"),
            Gross_Profit=("Gross Profit", "sum")
        )
    )

    regional["Profit_Margin_Percent"] = (
        regional["Gross_Profit"]
        / regional["Sales"].replace(0, np.nan)
        * 100
    ).fillna(0)

    st.dataframe(regional.round(2), width="stretch", hide_index=True)

    left, right = st.columns(2)

    with left:
        fig = px.bar(
            regional, x="Region", y="Sales",
            color="Region", title="Sales by Region",
            text_auto=".2s"
        )
        st.plotly_chart(fig, width="stretch")

    with right:
        fig = px.bar(
            regional, x="Region", y="Gross_Profit",
            color="Region", title="Gross Profit by Region",
            text_auto=".2s"
        )
        st.plotly_chart(fig, width="stretch")

    regional_shipping = (
        filtered_df.groupby(["Region", "Ship Mode"], as_index=False)
        .agg(
            Unique_Orders=("Order ID", "nunique"),
            Sales=("Sales", "sum"),
            Gross_Profit=("Gross Profit", "sum"),
            Units=("Units", "sum")
        )
    )

    st.subheader("Regional Shipping-Mode Performance")
    st.dataframe(
        regional_shipping.round(2),
        width="stretch",
        hide_index=True
    )


# ============================================================
# PAGE 4: PRODUCT INSIGHTS
# ============================================================

elif page == "Product Insights":
    st.header("🍫 Product Insights")

    if filtered_df.empty:
        show_empty()
        st.stop()

    summary = (
        filtered_df.groupby(
            ["Product ID", "Product Name"], as_index=False
        )
        .agg(
            Total_Units=("Units", "sum"),
            Total_Sales=("Sales", "sum"),
            Total_Cost=("Cost", "sum"),
            Total_Gross_Profit=("Gross Profit", "sum"),
            Order_Count=("Order ID", "nunique")
        )
    )

    summary["Profit_Margin_Percent"] = (
        summary["Total_Gross_Profit"]
        / summary["Total_Sales"].replace(0, np.nan)
        * 100
    ).fillna(0)

    summary = summary.sort_values(
        "Total_Gross_Profit", ascending=False
    )

    st.subheader("Product Profitability")
    st.dataframe(summary.round(2), width="stretch", hide_index=True)

    left, right = st.columns(2)

    with left:
        fig = px.bar(
            summary, x="Product Name", y="Total_Sales",
            color="Total_Sales", title="Sales by Product"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, width="stretch")

    with right:
        fig = px.bar(
            summary, x="Product Name", y="Total_Gross_Profit",
            color="Total_Gross_Profit", title="Gross Profit by Product"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Top 5 Products by Gross Profit")
    st.dataframe(summary.head(5).round(2), width="stretch", hide_index=True)

    st.subheader("5 Lowest Profit-Margin Products")
    st.dataframe(
        summary.sort_values("Profit_Margin_Percent").head(5).round(2),
        width="stretch",
        hide_index=True
    )


# ============================================================
# PAGE 5: PRODUCT SHIPPING ANALYSIS
# ============================================================

elif page == "Product Shipping Analysis":
    st.header("📦 Product Shipping Analysis")

    if filtered_df.empty:
        show_empty()
        st.stop()

    summary = (
        filtered_df.groupby(
            ["Product Name", "Ship Mode"], as_index=False
        )
        .agg(
            Unique_Orders=("Order ID", "nunique"),
            Units=("Units", "sum"),
            Sales=("Sales", "sum"),
            Cost=("Cost", "sum"),
            Gross_Profit=("Gross Profit", "sum")
        )
    )

    summary["Profit_Margin_Percent"] = (
        summary["Gross_Profit"]
        / summary["Sales"].replace(0, np.nan)
        * 100
    ).fillna(0)

    st.dataframe(summary.round(2), width="stretch", hide_index=True)

    metric = st.selectbox(
        "Choose metric",
        ["Sales", "Gross_Profit", "Units", "Unique_Orders"]
    )

    fig = px.bar(
        summary, x="Product Name", y=metric,
        color="Ship Mode", barmode="group",
        title=f"{metric.replace('_', ' ')} by Product and Shipping Mode"
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, width="stretch")

    st.info(
        "These are historical associations. Without actual freight "
        "charges, this comparison does not prove which shipping mode "
        "is cheaper."
    )


# ============================================================
# PAGE 6: RECOMMENDATION ENGINE
# ============================================================

elif page == "Recommendation Engine":
    st.header("🤖 Shipping Recommendation Engine")

    if filtered_df.empty:
        show_empty()
        st.stop()

    summary = shipping_summary(filtered_df)

    summary["Historical_Profit_Per_Order"] = (
        summary["Total_Gross_Profit"]
        / summary["Order_Count"].replace(0, np.nan)
    ).fillna(0)

    summary = summary.sort_values(
        "Historical_Profit_Per_Order", ascending=False
    )

    st.markdown(
        "This page ranks shipping modes using historical gross profit "
        "per unique order. It is a descriptive comparison, not a "
        "trained shipping optimization model."
    )

    st.dataframe(
        summary[
            [
                "Ship Mode", "Order_Count", "Total_Sales",
                "Total_Gross_Profit", "Profit_Margin_Percent",
                "Historical_Profit_Per_Order"
            ]
        ].round(2),
        width="stretch",
        hide_index=True
    )

    if not summary.empty:
        best = summary.iloc[0]
        st.success(
            f"Highest historical gross profit per order in this "
            f"selection: {best['Ship Mode']} "
            f"({money(best['Historical_Profit_Per_Order'])})."
        )

    st.warning(
        "Gross profit per order is not shipping-cost efficiency. "
        "Actual freight costs, service requirements and delivery "
        "performance are needed for a defensible recommendation."
    )

    st.subheader("Recommendation Checklist")
    st.markdown(
        """
        - Confirm the required delivery service.
        - Validate order and ship dates.
        - Obtain actual freight charges by shipping mode.
        - Compare delivery reliability and customer requirements.
        - Verify factory locations and capacities before considering
          factory reassignment.
        """
    )


# ============================================================
# PAGE 7: MACHINE LEARNING PREDICTIONS
# ============================================================

elif page == "ML Predictions":
    st.header("🤖 Gross Profit Prediction")
    st.caption(
        "Uses the saved preprocessing and prediction pipeline "
        "created by ml_models.py."
    )

    try:
        model = load_ml_model()
        model_results = load_model_results()
    except Exception as error:
        st.error(f"Unable to load the ML files: {error}")
        st.stop()

    if model is None:
        st.error(
            "best_profit_model.joblib was not found. "
            "Run ml_models.py first."
        )
        st.code("python ml_models.py", language="powershell")
        st.stop()

    if not model_results.empty:
        st.subheader("Model Comparison")

        st.dataframe(
            model_results.round(4),
            width="stretch",
            hide_index=True
        )

        if {"Model", "MAE", "RMSE", "R2 Score"}.issubset(
            model_results.columns
        ):
            best_result = model_results.sort_values("MAE").iloc[0]

            c1, c2, c3 = st.columns(3)
            c1.metric("Best Model", str(best_result["Model"]))
            c2.metric("Test MAE", f"{best_result['MAE']:.4f}")
            c3.metric("Test R²", f"{best_result['R2 Score']:.4f}")

            chart = px.bar(
                model_results,
                x="Model",
                y="MAE",
                color="Model",
                title="Model Comparison: Mean Absolute Error"
            )
            st.plotly_chart(chart, width="stretch")

    st.divider()
    st.subheader("Enter Prediction Details")

    # Use the original dataset to populate valid input choices.
    ship_modes = sorted(
        df["Ship Mode"].dropna().astype(str).unique().tolist()
    )
    prediction_regions = sorted(
        df["Region"].dropna().astype(str).unique().tolist()
    )
    divisions = sorted(
        df["Division"].dropna().astype(str).unique().tolist()
    )

    product_table = (
        df[["Product ID", "Product Name"]]
        .dropna()
        .drop_duplicates()
        .copy()
    )

    product_table["Product ID"] = product_table["Product ID"].astype(str)
    product_table["Product Name"] = product_table["Product Name"].astype(str)
    product_table = product_table.sort_values("Product Name")

    if (
        not ship_modes
        or not prediction_regions
        or not divisions
        or product_table.empty
    ):
        st.error("The dataset does not contain enough valid prediction inputs.")
        st.stop()

    left, right = st.columns(2)

    with left:
        chosen_mode = st.selectbox("Shipping Mode", ship_modes)
        chosen_region = st.selectbox("Region", prediction_regions)
        chosen_division = st.selectbox("Division", divisions)

    with right:
        product_options = (
            product_table["Product ID"]
            + " | "
            + product_table["Product Name"]
        ).tolist()

        chosen_product = st.selectbox("Product", product_options)
        product_index = product_options.index(chosen_product)
        product_row = product_table.iloc[product_index]

        chosen_units = st.number_input(
            "Units", min_value=1, value=1, step=1
        )

    if st.button("Predict Gross Profit", type="primary"):
        input_data = pd.DataFrame([{
            "Ship Mode": chosen_mode,
            "Region": chosen_region,
            "Division": chosen_division,
            "Product ID": product_row["Product ID"],
            "Product Name": product_row["Product Name"],
            "Units": chosen_units
        }])

        expected_features = [
            "Ship Mode", "Region", "Division",
            "Product ID", "Product Name", "Units"
        ]

        try:
            # Preserve the exact feature order used during training.
            input_data = input_data[expected_features]
            prediction = float(model.predict(input_data)[0])

            st.success(
                f"Predicted Gross Profit: {money(prediction)}"
            )

            st.caption(
                "This is an estimate from the trained model, not a "
                "guaranteed result. Model accuracy depends on the "
                "training data and may vary for new records."
            )

            output = input_data.copy()
            output["Predicted Gross Profit"] = prediction

            st.dataframe(output, width="stretch", hide_index=True)

            st.download_button(
                "Download Prediction CSV",
                data=output.to_csv(index=False).encode("utf-8"),
                file_name="gross_profit_prediction.csv",
                mime="text/csv"
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

    st.info(
        "This model predicts gross profit. It does not predict shipping "
        "duration or calculate shipping-cost savings."
    )


# ============================================================
# PAGE 8: WHAT-IF SIMULATOR
# ============================================================

elif page == "What-If Simulator":
    st.header("🧪 What-If Simulator")

    if filtered_df.empty:
        show_empty()
        st.stop()

    summary = shipping_summary(filtered_df)
    available_modes = summary["Ship Mode"].dropna().tolist()

    if not available_modes:
        st.warning("No shipping modes are available.")
        st.stop()

    reference_mode = st.selectbox(
        "Current shipping mode", available_modes, key="reference_mode"
    )

    comparison_mode = st.selectbox(
        "Comparison shipping mode",
        available_modes,
        index=1 if len(available_modes) > 1 else 0,
        key="comparison_mode"
    )

    simulated_orders = st.number_input(
        "Number of hypothetical orders",
        min_value=1, max_value=1000000, value=1000, step=100
    )

    reference = summary[summary["Ship Mode"] == reference_mode].iloc[0]
    comparison = summary[summary["Ship Mode"] == comparison_mode].iloc[0]

    ref_count = reference["Order_Count"]
    comp_count = comparison["Order_Count"]

    ref_per_order = (
        reference["Total_Gross_Profit"] / ref_count if ref_count else 0
    )
    comp_per_order = (
        comparison["Total_Gross_Profit"] / comp_count if comp_count else 0
    )

    ref_profit = ref_per_order * simulated_orders
    comp_profit = comp_per_order * simulated_orders
    difference = comp_profit - ref_profit

    c1, c2, c3 = st.columns(3)
    c1.metric("Reference Historical GP", money(ref_profit))
    c2.metric("Comparison Historical GP", money(comp_profit))
    c3.metric("Difference", money(difference))

    st.info(
        f"Illustrative comparison of historical gross profit per unique "
        f"order for {reference_mode} and {comparison_mode}, applied to "
        f"{simulated_orders:,.0f} hypothetical orders."
    )

    st.warning(
        "This is not a shipping-cost savings estimate, causal forecast, "
        "or ML prediction. It assumes historical averages remain "
        "applicable and excludes freight charges and service constraints."
    )


# ============================================================
# PAGE 9: DATA QUALITY AND RISK
# ============================================================

elif page == "Data Quality & Risk":
    st.header("🔎 Data Quality & Risk")

    full = df.copy()

    checks = [
        ("Total dataset rows", len(full)),
        ("Duplicate rows", int(full.duplicated().sum())),
        ("Missing order dates", int(full["Order Date"].isna().sum())),
        ("Missing ship dates", int(full["Ship Date"].isna().sum())),
        ("Missing sales values", int(full["Sales"].isna().sum())),
        ("Missing cost values", int(full["Cost"].isna().sum())),
        ("Missing gross profit values", int(full["Gross Profit"].isna().sum())),
        (
            "Negative lead times",
            int((full["Shipping_Lead_Time"] < 0).sum())
        ),
        (
            "Shipping durations over 30 days",
            int((full["Shipping_Lead_Time"] > 30).sum())
        ),
        ("Missing Order IDs", int(full["Order ID"].isna().sum()))
    ]

    st.subheader("Data Quality Checks")
    st.dataframe(
        pd.DataFrame(checks, columns=["Check", "Result"]),
        width="stretch",
        hide_index=True
    )

    st.subheader("Date Ranges")
    left, right = st.columns(2)

    with left:
        st.markdown("**Order Date Range**")
        if full["Order Date"].notna().any():
            st.write(
                f"{full['Order Date'].min().date()} to "
                f"{full['Order Date'].max().date()}"
            )
        else:
            st.write("No valid order dates")

    with right:
        st.markdown("**Ship Date Range**")
        if full["Ship Date"].notna().any():
            st.write(
                f"{full['Ship Date'].min().date()} to "
                f"{full['Ship Date'].max().date()}"
            )
        else:
            st.write("No valid ship dates")

    st.subheader("Lead-Time Statistics")
    lead_times = full["Shipping_Lead_Time"].dropna()

    if not lead_times.empty:
        st.dataframe(
            lead_times.describe().to_frame("Lead Time in Days"),
            width="stretch"
        )

    st.subheader("Shipping Duration by Mode")

    lead_by_mode = (
        full.dropna(subset=["Shipping_Lead_Time", "Ship Mode"])
        .groupby("Ship Mode")["Shipping_Lead_Time"]
        .agg(
            Record_Count="count",
            Minimum_Days="min",
            Median_Days="median",
            Maximum_Days="max",
            Average_Days="mean"
        )
        .reset_index()
    )

    st.dataframe(
        lead_by_mode.round(2),
        width="stretch",
        hide_index=True
    )

    st.subheader("Order ID Year Audit")
    st.write(
        "The year embedded in an Order ID is an audit clue, not proof "
        "that the order date is incorrect."
    )

    id_year_audit = (
        full.groupby("ID Year", dropna=False)
        .agg(
            Records=("Order ID", "size"),
            Earliest_Order_Date=("Order Date", "min"),
            Latest_Order_Date=("Order Date", "max")
        )
        .reset_index()
    )

    st.dataframe(
        id_year_audit,
        width="stretch",
        hide_index=True
    )

    st.subheader("⚠️ Important Data Risks")

    if full["Order Date"].notna().any() and full["Ship Date"].notna().any():
        earliest_order = full["Order Date"].min().date()
        latest_order = full["Order Date"].max().date()
        earliest_ship = full["Ship Date"].min().date()
        latest_ship = full["Ship Date"].max().date()

        st.warning(
            f"Order dates range from {earliest_order} to {latest_order}; "
            f"ship dates range from {earliest_ship} to {latest_ship}. "
            "Investigate unusually long durations before using them."
        )

    st.markdown(
        """
        **Recommended handling**

        1. Preserve the original dates.
        2. Investigate discrepancies with the data source.
        3. Do not use unvalidated shipping durations as a reliable
           machine-learning target.
        4. Do not invent replacement dates.
        5. Actual freight costs, factory assignments, factory capacity
           and route distances are needed for defensible shipping-cost
           and factory-reallocation optimization.
        """
    )

    st.info(
        "The 30-day threshold is an investigation flag, not a verified "
        "business rule. A flagged row is not automatically incorrect."
    )

    st.subheader("Profit Consistency Check")

    profit_difference = (
        full["Sales"] - full["Cost"] - full["Gross Profit"]
    ).abs()

    inconsistent = int((profit_difference > 0.01).sum())

    st.metric(
        "Rows where Sales − Cost differs from Gross Profit",
        inconsistent
    )

    if inconsistent == 0:
        st.success(
            "Sales minus Cost matches Gross Profit within a 0.01 tolerance."
        )
    else:
        st.warning(
            "Some rows have inconsistent sales, cost and gross profit. "
            "Review them before modelling."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "Nassau Candy Distributor | Shipping Optimization Recommendation System"
)