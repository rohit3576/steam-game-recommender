steam-game-recommender/
│
├── app.py # Streamlit UI dashboard
├── train_model.py # ML training script
├── games.csv # Dataset (Steam games)
├── model.pkl # Trained ML model
├── features.pkl # Features list used in training
├── requirements.txt # Project dependencies
└── README.md # Project documentation


---

## ⚙️ Installation & Setup

### ✅ 1) Clone the Repository
```bash
git clone https://github.com/rohit3576/steam-game-recommender.git
cd steam-game-recommender

✅ 2) Create & Activate Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate

Mac/Linux
python3 -m venv venv
source venv/bin/activate

✅ 3) Install Dependencies
pip install -r requirements.txt

🧪 Train the Model

Run the training script to generate:

✅ model.pkl
✅ features.pkl

python train_model.py

🎮 Run the Streamlit App
streamlit run app.py

📊 Example Output

✅ Prediction Output:

HIT GAME DETECTED 🎮
or

NICHE MARKET 📊

✅ Confidence Meter shows probability score
✅ Market comparison graphs update dynamically

📌 Future Improvements (Planned)

🚀 Add a Steam Game Recommendation System (TF-IDF + Cosine Similarity)
📈 Add more ML models & hyperparameter tuning
🎯 Add filters like Genre, Tags, Multiplayer, etc.
🌐 Deploy using Streamlit Cloud / Render

👨‍💻 Author

Rohit Pawar
GitHub: @rohit3576