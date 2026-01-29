import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier

# Load
df = pd.read_csv("games.csv")

# ✅ Target (label): Hit or Not
df["hit"] = ((df["positive_ratio"] >= 85) & (df["user_reviews"] >= 500)).astype(int)

# ✅ Features (inputs)
features = ["price_final", "discount", "win", "mac", "linux", "steam_deck", "user_reviews"]
X = df[features]
y = df["hit"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Results
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n✅ Report:\n", classification_report(y_test, y_pred))
