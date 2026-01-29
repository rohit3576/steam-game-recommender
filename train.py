import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("games.csv")

# ✅ Create target: HIT game or not
# Hit = good rating + enough reviews
df["hit"] = ((df["positive_ratio"] >= 85) & (df["user_reviews"] >= 500)).astype(int)

# ✅ Features for model
features = ["price_final", "discount", "win", "mac", "linux", "steam_deck", "user_reviews"]
X = df[features]
y = df["hit"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, pred))

# Save model + features
joblib.dump(model, "model.pkl")
joblib.dump(features, "features.pkl")

print("✅ Saved model.pkl and features.pkl")
