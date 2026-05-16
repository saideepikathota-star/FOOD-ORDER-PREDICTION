import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Page config ---
st.set_page_config(page_title="NextBite - Food Order Prediction", page_icon="🍽️", layout="wide")

# --- Header ---
st.title("🍽️ NextBite - Next Food Item Prediction")
st.markdown("""
Predict the **top 3 items** a customer is likely to order next.  
The most probable item is highlighted with probability bars.
""")

# --- Load model, encoders, scaler, and dataset ---
MODEL_PATH = "final_model.pkl"  # Make sure this file is in your GitHub repo
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please ensure 'final_model.pkl' is in the repo.")

@st.cache_data
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load("label_encoders.pkl")
    scaler = joblib.load("scaler.pkl")
    df = pd.read_csv("food_orders_clean.csv")
    return model, encoders, scaler, df

best_model, label_encoders, scaler, df = load_artifacts()

# --- Sidebar Inputs ---
st.sidebar.header("Enter Customer Order Details")
add_to_cart_order = st.sidebar.number_input("Add to Cart Order (1–10)", min_value=1, max_value=50, value=1)
reordered = st.sidebar.selectbox("Reordered (0 = No, 1 = Yes)", [0, 1])
order_number = st.sidebar.number_input("Order Number", min_value=1, value=1)
order_dow = st.sidebar.selectbox("Order Day of Week (0=Sunday ... 6=Saturday)", range(7))
order_hour_of_day = st.sidebar.slider("Order Hour of Day (0–23)", 0, 23)
days_since_prior_order = st.sidebar.number_input("Days Since Prior Order", min_value=0, value=0)
aisle = st.sidebar.selectbox("Aisle", options=label_encoders["aisle"].classes_)
department = st.sidebar.selectbox("Department", options=label_encoders["department"].classes_)
age = st.sidebar.number_input("Customer Age", min_value=1, max_value=120, value=25)
gender = st.sidebar.selectbox("Gender", options=label_encoders["gender"].classes_)
location = st.sidebar.selectbox("Location", options=label_encoders["location"].classes_)

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["Prediction", "EDA / Insights", "About Project"])

# --- Prediction Tab ---
with tab1:
    st.subheader("Single Customer Prediction")

    def predict_top3():
        sample_input = {
            "add_to_cart_order": add_to_cart_order,
            "reordered": reordered,
            "order_number": order_number,
            "order_dow": order_dow,
            "order_hour_of_day": order_hour_of_day,
            "days_since_prior_order": days_since_prior_order,
            "aisle": aisle,
            "department": department,
            "age": age,
            "gender": gender,
            "location": location,
            "order_id": 0,
            "product_id": 0,
            "user_id": 0
        }

        sample_df = pd.DataFrame([sample_input])

        # Encode categorical features
        for col in ["aisle", "department", "gender", "location"]:
            le = label_encoders[col]
            if sample_input[col] in le.classes_:
                sample_df[col] = le.transform([sample_input[col]])[0]
            else:
                sample_df[col] = df[col].mode()[0]

        # Reorder columns exactly as scaler expects
        sample_df = sample_df[scaler.feature_names_in_]

        # Scale features
        sample_scaled = scaler.transform(sample_df)

        # Predict probabilities
        probs = best_model.predict_proba(sample_scaled)[0]

        # Decode product names
        product_encoder = label_encoders["product_name"]
        food_items = product_encoder.inverse_transform(np.arange(len(probs)))

        predictions = pd.DataFrame({
            "Food Item": food_items,
            "Probability": probs
        }).sort_values(by="Probability", ascending=False)

        return predictions.head(3)

    if st.button("Predict Next Items"):
        top3 = predict_top3()

        # Side-by-side cards
        st.subheader("Top 3 Predicted Items")
        col1, col2, col3 = st.columns(3)
        columns = [col1, col2, col3]

        for i, (idx, row) in enumerate(top3.iterrows()):
            with columns[i]:
                color = "#85C1E9" if i == 0 else "#AED6F1"
                st.markdown(f"""
                <div style="border:2px solid {color}; padding:20px; border-radius:15px; background-color:#E8F6F3; text-align:center">
                    <h4 style="color:#1B4F72;">{row['Food Item']}</h4>
                    <div style="background-color:#D6EAF8; width:100%; border-radius:5px">
                        <div style="width:{row['Probability']*100:.2f}%; background-color:#1F618D; color:white; padding:5px; border-radius:5px">
                            {row['Probability']*100:.2f}%
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        st.success(f"Most likely food item: **{top3.iloc[0]['Food Item']}**")

# --- EDA Tab ---
with tab2:
    st.subheader("EDA and Insights")
    st.markdown("Explore dataset trends and popular items.")

    # Filter by location
    filter_location = st.selectbox("Filter by Location", options=["All"] + list(df['location'].unique()))
    df_filtered = df.copy()
    if filter_location != "All":
        df_filtered = df_filtered[df_filtered['location'] == filter_location]

    st.markdown("**Top 10 Most Popular Food Items:**")
    st.bar_chart(df_filtered['product_name'].value_counts().head(10))

    st.markdown("**Orders by Hour of Day:**")
    st.bar_chart(df_filtered['order_hour_of_day'].value_counts().sort_index())

    st.markdown("**Orders by Day of Week:**")
    st.bar_chart(df_filtered['order_dow'].value_counts().sort_index())

# --- About Tab ---
with tab3:
    st.subheader("About NextBite Project")
    st.markdown("""
- **Goal:** Predict the next items a customer is likely to order using historical data.
- **Dataset:** `food_orders_clean.csv` with features like `product_name`, `aisle`, `department`, `age`, `gender`, `location`, etc.
- **ML Pipeline:** Preprocessing → Encoding categorical variables → Scaling numerical features → Multi-class classification → Top 3 item prediction.
- **Features Used:** Numeric and categorical features including order info and customer demographics.
- **Evaluation Metrics:** Accuracy, Top-K accuracy, probability scores.
- **Academic Value:** Demonstrates end-to-end ML pipeline, interactive deployment, EDA, and interpretable results.
""")
