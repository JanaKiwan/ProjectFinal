import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load data function with caching
@st.cache_data
def load_data(uploaded_file):
    try:
        return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        return None
# App Title
st.title("Enhanced Customer Segmentation & Insights Dashboard")
st.write("""
**Analyze customer behavior, segment trends, and key purchase metrics with detailed insights.**
""")
# File uploader
uploaded_file = st.sidebar.file_uploader("Upload Excel File:", type=["xlsx"])
if uploaded_file is not None:
    # Load the data
    data = load_data(uploaded_file)
    if data is not None:
        # Sidebar Navigation
        st.sidebar.title("Navigation")
        section = st.sidebar.radio("Go To:", [
            "Overview", 
            "Key Insights", 
            "Interactive Filtering", 
            "Top Customers", 
            "Feature Comparisons", 
            "Correlation Analysis",
            "Segment Exploration"
        ])
        # Section: Overview
        if section == "Overview":
            st.header("Dataset Overview")
            st.write("### Quick Summary:")
            st.dataframe(data.head())
            st.write(f"**Total Customers:** {data.shape[0]}")
            st.write(f"**Features Available:** {data.shape[1]}")
            st.subheader("Key Metrics:")
            st.metric("Mean CLTV", f"${data['CLTV'].mean():,.2f}")
            st.metric("Mean Purchase Probability", f"{data['Purchase Probability'].mean() * 100:.2f}%")
            st.metric("Top Purchaser CLTV", f"${data['CLTV'].max():,.2f}")
            # Histogram of CLTV
            st.subheader("CLTV Distribution")
            fig, ax = plt.subplots()
            sns.histplot(data['CLTV'], bins=20, kde=True, ax=ax)
            ax.set_title("CLTV Distribution Across Customers")
            st.pyplot(fig)
        # Section: Key Insights
        elif section == "Key Insights":
            st.header("Key Insights")
            st.write("### What defines a top purchaser?")
            top_purchasers = data[data['CLTV'] > data['CLTV'].quantile(0.9)]
            st.write(f"**Top Purchasers (90th Percentile CLTV):** {top_purchasers.shape[0]} customers.")
            st.write(f"Average Purchase Probability: **{top_purchasers['Purchase Probability'].mean() * 100:.2f}%**")
            st.write(f"Average Refund Ratio: **{top_purchasers['Refund_Ratio'].mean() * 100:.2f}%**")
            st.write(f"Frequent Item Group: **{top_purchasers['Most Frequent Item_Group'].mode()[0]}**")
            st.write(f"Common Trend: **{top_purchasers['Most Frequent Trend_Classification'].mode()[0]}**")
            # Histogram for top purchasers
            st.subheader("CLTV Distribution for Top Purchasers")
            fig, ax = plt.subplots()
            sns.histplot(top_purchasers['CLTV'], bins=20, kde=True, ax=ax, color='green')
            ax.set_title("CLTV Distribution (Top Purchasers)")
            st.pyplot(fig)
        # Section: Interactive Filtering
        elif section == "Interactive Filtering":
            st.header("Interactive Filtering")
            # Sliders for filtering
            min_cltv = st.slider("Minimum CLTV", 0, int(data['CLTV'].max()), 0)
            max_cltv = st.slider("Maximum CLTV", 0, int(data['CLTV'].max()), int(data['CLTV'].max()))
            min_prob = st.slider("Minimum Purchase Probability (%)", 0, 100, 0) / 100
            filtered_data = data[
                (data['CLTV'] >= min_cltv) & 
                (data['CLTV'] <= max_cltv) & 
                (data['Purchase Probability'] >= min_prob)
            ]
            st.write(f"### Filtered Data ({filtered_data.shape[0]} customers):")
            st.dataframe(filtered_data)
            st.write("### Summary of Filtered Data:")
            st.write(filtered_data.describe())
        # Section: Top Customers
        elif section == "Top Customers":
            st.header("Top Customers")
            # Display Top 10 Purchasers
            st.write("### Top 10 Customers by CLTV")
            top_customers = data.nlargest(10, 'CLTV')
            st.dataframe(top_customers[['CUSTOMERNAME', 'CLTV', 'Purchase Probability', 'Segment']])
            # Export Top Customers
            st.download_button(
                label="Download Top Customers as CSV",
                data=top_customers.to_csv(index=False),
                file_name="top_customers.csv",
                mime="text/csv"
            )
        # Section: Feature Comparisons
        elif section == "Feature Comparisons":
            st.header("Feature Comparisons")
            st.write("### Compare Features Across Customer Types:")
            low_value = data[data['Segment'] == 'Low-Engagement New Customers']
            high_value = data[data['Segment'] == 'Consistent High-Frequency Spenders']
            col1, col2 = st.columns(2)
            with col1:
                st.write("#### Low-Engagement Customers")
                st.metric("Mean CLTV", f"${low_value['CLTV'].mean():,.2f}")
                st.metric("Mean Purchase Frequency", f"{low_value['Purchase_Frequency_Per_Month'].mean():.2f}")
                st.metric("Refund Ratio", f"{low_value['Refund_Ratio'].mean() * 100:.2f}%")
            with col2:
                st.write("#### High-Frequency Spenders")
                st.metric("Mean CLTV", f"${high_value['CLTV'].mean():,.2f}")
                st.metric("Mean Purchase Frequency", f"{high_value['Purchase_Frequency_Per_Month'].mean():.2f}")
                st.metric("Refund Ratio", f"{high_value['Refund_Ratio'].mean() * 100:.2f}%")
            # Comparison Scatterplot
            st.write("### CLTV vs Purchase Frequency (Segmented)")
            fig, ax = plt.subplots()
            sns.scatterplot(x='Purchase_Frequency_Per_Month', y='CLTV', hue='Segment', data=data, ax=ax)
            ax.set_title("CLTV vs Purchase Frequency")
            st.pyplot(fig)
        # Section: Correlation Analysis
        elif section == "Correlation Analysis":
            st.header("Correlation Analysis")      
            st.write("### Heatmap of Feature Correlations:")
            corr = data.corr()
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        # Section: Segment Exploration
        elif section == "Segment Exploration":
            st.header("Explore Customer Segments")
            segment = st.selectbox("Select Segment to Explore:", data['Segment'].unique())
            segment_data = data[data['Segment'] == segment]
            st.write(f"### {segment} Segment Analysis:")
            st.dataframe(segment_data)
            st.write("### Segment CLTV Distribution:")
            fig, ax = plt.subplots()
            sns.histplot(segment_data['CLTV'], bins=20, kde=True, ax=ax)
            ax.set_title(f"CLTV Distribution for {segment}")
            st.pyplot(fig)
else:
    st.info("Please upload an Excel file to proceed.")