import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
@st.cache_data
def load_data():
    return pd.read_excel("customer_data_with_cltv.xlsx")

data = load_data()

# Define navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to", 
    ["Overview", "Feature Insights", "Segmentation", "Model Results", "Customer Drill-Down", "Item Group Analysis", "High-Value Customers"]
)

# Page 1: Overview
if page == "Overview":
    st.title("Customer Purchase Prediction: Overview")
    st.markdown("**Explore customer purchase probabilities, CLTV, and segmentation insights in this interactive dashboard.**")

    # Sidebar Filters
    st.sidebar.header("Filters")
    customer_segment = st.sidebar.multiselect(
        "Select Customer Segment",
        options=data["Segment"].unique(),
        default=data["Segment"].unique()
    )
    customer_lifetime_category = st.sidebar.multiselect(
        "Select Lifetime Category",
        options=data["Customer_Lifetime_Category"].unique(),
        default=data["Customer_Lifetime_Category"].unique()
    )

    # Filtered data
    filtered_data = data[
        (data["Segment"].isin(customer_segment)) & 
        (data["Customer_Lifetime_Category"].isin(customer_lifetime_category))
    ]

    st.subheader("Summary Metrics")
    st.write(f"**Total Customers:** {filtered_data['CUSTOMERNAME'].nunique()}")
    st.write(f"**Average CLTV:** AED {filtered_data['CLTV'].mean():,.2f}")
    st.write(f"**Average Purchase Probability:** {filtered_data['Purchase Probability'].mean():.2f}")

    # Visualizations
    st.subheader("Visualizations")
    st.write("Explore customer purchase probabilities and CLTV below:")

    # Histogram of Purchase Probability
    if st.checkbox("Show Purchase Probability Distribution"):
        fig, ax = plt.subplots()
        sns.histplot(filtered_data["Purchase Probability"], kde=True, ax=ax)
        ax.set_title("Distribution of Purchase Probability")
        ax.set_xlabel("Purchase Probability")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    # CLTV vs. Purchase Probability
    if st.checkbox("Show CLTV vs Purchase Probability"):
        fig, ax = plt.subplots()
        sns.scatterplot(
            data=filtered_data,
            x="Purchase Probability",
            y="CLTV",
            hue="Customer_Lifetime_Category",
            palette="viridis",
            ax=ax
        )
        ax.set_title("CLTV vs Purchase Probability by Lifetime Category")
        ax.set_xlabel("Purchase Probability")
        ax.set_ylabel("CLTV")
        st.pyplot(fig)

# Page 2: Feature Insights
elif page == "Feature Insights":
    st.title("Feature Insights")
    st.markdown("Explore key factors influencing customer purchase probability and CLTV.")
    
    # Feature Importance Barplot
    st.subheader("Feature Importance for Predicting Purchase Probability")
    feature_importance = {
        "Customer Lifetime": 0.85,
        "Average Purchase Per Month": -0.75,
        "Total Volume Purchased": 0.65,
        "Net Purchases": 0.60,
        "Maximum Time Without Purchase": 0.45,
        "Refund Ratio": -0.30,
    }
    fig, ax = plt.subplots()
    sns.barplot(
        x=list(feature_importance.values()), 
        y=list(feature_importance.keys()), 
        palette="coolwarm", 
        ax=ax
    )
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance")
    st.pyplot(fig)

# Page 3: Segmentation
elif page == "Segmentation":
    st.title("Customer Segmentation Insights")
    st.markdown("Explore customer lifetime categories and segmentation metrics.")

    # Segmentation Insights
    st.subheader("Customer Lifetime Distribution")
    lifetime_dist = data["Customer_Lifetime_Category"].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=lifetime_dist.index, y=lifetime_dist.values, palette="coolwarm", ax=ax)
    ax.set_title("Distribution of Customer Lifetime Categories")
    ax.set_xlabel("Customer Lifetime Category")
    ax.set_ylabel("Number of Customers")
    st.pyplot(fig)

    st.subheader("Segmentation Metrics")
    segment_metrics = data.groupby("Customer_Lifetime_Category").agg({
        "CLTV": ["mean", "sum"],
        "Purchase Probability": "mean",
        "Refund_Ratio": "mean",
        "Total Amount Purchased": "sum"
    }).reset_index()
    segment_metrics.columns = ["Customer Lifetime Category", "Avg CLTV", "Total CLTV", "Avg Purchase Probability", "Avg Refund Ratio", "Total Amount Purchased"]

    st.dataframe(segment_metrics)

    # High Refund Ratio Segments
    st.subheader("High Refund Ratio Analysis")
    high_refund = segment_metrics[segment_metrics["Avg Refund Ratio"] > 0.2]
    if not high_refund.empty:
        st.markdown("Segments with a high refund ratio (greater than 20%):")
        st.dataframe(high_refund)
    else:
        st.markdown("No segments with a refund ratio higher than 20% found.")

    # Filter and export data by segment
    st.subheader("Filter Customers by Lifetime Category")
    selected_segment = st.selectbox("Select Customer Lifetime Category", options=data["Customer_Lifetime_Category"].unique())
    filtered_segment_data = data[data["Customer_Lifetime_Category"] == selected_segment]

    st.write(f"Displaying data for {selected_segment} category:")
    st.dataframe(filtered_segment_data)

    # Export filtered data
    st.download_button(
        label="Download Filtered Segment Data",
        data=filtered_segment_data.to_csv(index=False),
        file_name=f"{selected_segment}_customers.csv"
    )

# Page 4: Predictive Modeling Results
elif page == "Model Results":
    st.title("Predictive Modeling Results")
    st.markdown("Review performance metrics for models predicting customer purchase probability.")

    st.subheader("Model Performance Comparison")
    model_results = pd.DataFrame({
        "Model": ["Lasso", "Logistic Regression", "SVC", "Decision Tree", "CatBoost"],
        "Validation Accuracy": [0.83, 0.82, 0.82, 0.75, 0.86],
        "Test Accuracy": [0.84, 0.87, 0.84, 0.83, 0.81],
        "Recall": [0.88, 0.83, 0.85, 0.93, 0.77],
        "Precision": [0.90, 0.91, 0.90, 0.83, 0.93],
        "Specificity": [0.81, 0.85, 0.83, 0.65, 0.89],
        "AUC": [0.93, 0.93, 0.93, 0.89, 0.90],
    })
    st.dataframe(model_results)

    st.subheader("Detailed Metrics for Models")
    detailed_metrics = pd.DataFrame({
        "Model": ["Lasso", "Logistic Regression", "SVC", "Decision Tree", "CatBoost"],
        "F1 Score": [0.89, 0.87, 0.88, 0.88, 0.84],
        "AUC": [0.93, 0.93, 0.93, 0.89, 0.90],
        "True Positives (TP)": [121, 114, 117, 127, 157],
        "False Negatives (FN)": [16, 23, 20, 10, 48],
        "False Positives (FP)": [14, 11, 13, 26, 12],
        "True Negatives (TN)": [61, 64, 62, 49, 101],
    })
    st.dataframe(detailed_metrics)
# Page 5: Customer Drill-Down
elif page == "Customer Drill-Down":
    st.title("Customer Drill-Down")
    st.markdown("Analyze individual customer details and purchase predictions.")

    # Customer Selection
    customer_name = st.selectbox("Select a Customer", options=data["CUSTOMERNAME"].unique())
    customer_data = data[data["CUSTOMERNAME"] == customer_name]

    st.subheader(f"Details for: {customer_name}")
    st.write(customer_data.T.rename(columns={customer_data.index[0]: "Value"}))

    # Export Single Customer Data
    st.download_button(
        label="Download Customer Data",
        data=customer_data.to_csv(index=False),
        file_name=f"{customer_name}_details.csv"
    )

# Page 6: Item Group Analysis
elif page == "Item Group Analysis":
    st.title("Item Group Analysis")
    st.markdown("Insights on item groups with high purchase probabilities and refunds.")

    # Refund Analysis
    st.subheader("Refund Analysis by Item Group")
    refund_data = data.groupby("ITEMGROUP").agg({"Refund_Ratio": "mean"}).reset_index()
    fig, ax = plt.subplots()
    sns.barplot(data=refund_data, x="Refund_Ratio", y="ITEMGROUP", ax=ax, palette="coolwarm")
    ax.set_title("Average Refund Ratios by Item Group")
    ax.set_xlabel("Refund Ratio")
    ax.set_ylabel("Item Group")
    st.pyplot(fig)

    # Purchase Probability by Item Group
    st.subheader("Purchase Probability by Item Group")
    prob_data = data.groupby("ITEMGROUP").agg({"Purchase Probability": "mean"}).reset_index()
    fig, ax = plt.subplots()
    sns.barplot(data=prob_data, x="Purchase Probability", y="ITEMGROUP", ax=ax, palette="viridis")
    ax.set_title("Average Purchase Probability by Item Group")
    ax.set_xlabel("Purchase Probability")
    ax.set_ylabel("Item Group")
    st.pyplot(fig)

    # Top Refunded Item Groups
    st.subheader("Top Refunded Item Groups")
    top_refunded_items = refund_data.nlargest(10, "Refund_Ratio")
    st.dataframe(top_refunded_items)

# Page 7: High-Value Customers
elif page == "High-Value Customers":
    st.title("High-Value Customers")
    st.markdown("Explore features of customers with the highest purchase amounts.")

    # Top 10 High-Value Customers
    top_customers = data.nlargest(10, "Total Amount Purchased")
    st.subheader("Top 10 High-Value Customers")
    st.dataframe(top_customers[["CUSTOMERNAME", "Total Amount Purchased", "Purchase Probability", "CLTV"]])

    # Features of High-Value Customers
    st.subheader("Features of High-Value Customers")
    high_value_features = top_customers[[
        "CUSTOMERNAME", 
        "Purchase Frequency Per Month", 
        "Maximum Time Without Purchase", 
        "Refund_Ratio", 
        "CLTV"
    ]]
    st.dataframe(high_value_features)

    # High-Value Customer Insights
    st.subheader("Key Insights")
    st.markdown(f"""
        - **Average CLTV of Top 10 Customers:** AED {top_customers['CLTV'].mean():,.2f}
        - **Average Purchase Frequency per Month:** {top_customers['Purchase Frequency Per Month'].mean():.2f}
        - **Average Refund Ratio:** {top_customers['Refund_Ratio'].mean():.2f}
    """)

    # Export Top Customers
    st.download_button(
        label="Download High-Value Customers Data",
        data=top_customers.to_csv(index=False),
        file_name="high_value_customers.csv"
    )
