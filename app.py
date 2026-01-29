import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Steam Game Recommender", layout="centered")

st.title("🎮 Steam Game Recommender + Hit Predictor")
st.write("Select a game and see if it will be a **Hit** (Highly Recommended) or not.")

# Load data
df = pd.read_csv("games.csv")

# Load model
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

# Dropdown to choose a game
game_name = st.selectbox("Choose a game:", df["title"].unique())

# Get selected game row
game = df[df["title"] == game_name].iloc[0]

st.subheader("📌 Game Info")
st.write(f"⭐ Rating: **{game['rating']}**")
st.write(f"✅ Positive Ratio: **{game['positive_ratio']}**")
st.write(f"🧾 Reviews: **{game['user_reviews']}**")
st.write(f"💰 Final Price: **{game['price_final']}**")
st.write(f"🔻 Discount: **{game['discount']}%**")

# Predict button
if st.button("🔍 Predict Hit or Not"):
    input_data = np.array([[game[f] for f in features]])
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if pred == 1:
        st.success(f"🔥 HIT Game! Probability: {prob:.2f}")
    else:
        st.warning(f"😅 Not a Hit (Normal). Probability: {prob:.2f}")
