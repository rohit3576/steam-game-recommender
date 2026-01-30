# 🎮 Steam Game Success Predictor

A machine learning-powered web application that analyzes Steam game metrics and predicts market success with AI-powered insights.

## 📋 Project Overview

This project uses machine learning to analyze Steam game data and predict whether a game will be a market hit or have niche appeal. The application features an interactive dashboard with real-time predictions, visual analytics, and market comparisons.

## 🏗️ Project Structure

```
steam-game-recommender/
│
├── app.py                    # Streamlit UI dashboard
├── train_model.py            # ML training script
├── games.csv                 # Dataset (Steam games)
├── model.pkl                 # Trained ML model
├── features.pkl              # Features list used in training
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## 🚀 Features

- **🎯 Success Prediction**: AI-powered prediction of game success (Hit Game vs Niche Market)
- **📊 Real-time Analytics**: Interactive charts and metrics comparison
- **🎮 Gaming UI**: Dark theme with neon accents and gaming aesthetics
- **📈 Market Insights**: Detailed statistics and market position analysis
- **🔍 Feature Importance**: Visual representation of key success factors
- **📱 Responsive Design**: Works on desktop and mobile devices

## ⚙️ Installation & Setup

### ✅ 1. Clone the Repository
```bash
git clone https://github.com/rohit3576/steam-game-recommender.git
cd steam-game-recommender
```

### ✅ 2. Create & Activate Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### ✅ 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 🧪 Train the Model

Run the training script to generate the ML model and features:

```bash
python train_model.py
```

This will create:
- ✅ `model.pkl` - Trained ML model
- ✅ `features.pkl` - Features list used in training

## 🎮 Run the Streamlit App

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📊 How It Works

### 1. **Data Input**
- Select any game from the Steam database in the sidebar
- Real-time metrics display for the selected game

### 2. **Prediction Analysis**
- Click "Run Prediction Analysis" to get AI-powered success prediction
- Results show either "HIT GAME DETECTED" 🎮 or "NICHE MARKET" 📊
- Confidence score with visual meter

### 3. **Visual Analytics**
- **Dashboard**: Key metrics and market comparison
- **Analysis**: Feature importance and game profile
- **Insights**: Market statistics and detailed game data

### 4. **Output Examples**
```
✅ HIT GAME DETECTED! (85.3% confidence)
High probability of commercial success

✅ NICHE MARKET (42.1% confidence)
Standard market performance expected
```

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Backend**: Python, Scikit-learn, Pandas, NumPy
- **Visualization**: Plotly, Matplotlib
- **ML Algorithms**: Classification models
- **Styling**: Custom CSS with gaming theme

## 📁 Dataset Information

The `games.csv` file contains Steam game data including:
- Game titles and release information
- Pricing and discount details
- User reviews and ratings
- Platform support (Windows/Mac/Linux)
- Various game metrics and features

## 🔧 Model Details

The machine learning model uses features such as:
- Price and discount percentage
- User review count and positive ratio
- Platform availability
- Release year and age
- Other relevant game metrics

## 🎨 UI Features

- **Dark Gaming Theme**: Cyberpunk-inspired design with neon accents
- **Interactive Charts**: Real-time data visualization
- **Responsive Layout**: Adapts to different screen sizes
- **Animated Elements**: Hover effects and transitions
- **Custom Components**: Gaming-style badges and cards

## 📌 Future Improvements (Planned)

### 🚀 Phase 1 - Enhanced Features
- [ ] **Steam Game Recommendation System** (TF-IDF + Cosine Similarity)
- [ ] **Advanced Filtering** (Genre, Tags, Multiplayer, Release Date)
- [ ] **User Preferences** (Personalized recommendations)

### 📈 Phase 2 - ML Enhancements
- [ ] **Multiple ML Models** (Random Forest, XGBoost, Neural Networks)
- [ ] **Hyperparameter Tuning** (Grid Search, Bayesian Optimization)
- [ ] **Feature Engineering** (Additional derived features)

### 🌐 Phase 3 - Deployment & Scale
- [ ] **Cloud Deployment** (Streamlit Cloud / Render / AWS)
- [ ] **Database Integration** (PostgreSQL / MongoDB)
- [ ] **API Development** (REST API for predictions)
- [ ] **User Authentication** (Login system with profiles)

### 📊 Phase 4 - Analytics & Insights
- [ ] **Trend Analysis** (Market trends over time)
- [ ] **Competitor Analysis** (Similar game comparisons)
- [ ] **Price Optimization** (Optimal pricing suggestions)
- [ ] **Revenue Prediction** (Estimated sales projections)

### 🎮 Phase 5 - Gaming Community Features
- [ ] **User Reviews Integration** (Steam API integration)
- [ ] **Community Ratings** (User voting system)
- [ ] **Game Collections** (Create and share game lists)
- [ ] **Social Features** (Share predictions, follow users)

## 🐛 Troubleshooting

### Common Issues & Solutions:

**1. "Required files not found" error**
```bash
# Ensure all files are in the correct directory:
# - games.csv
# - model.pkl  
# - features.pkl
# Run the training script if files are missing:
python train_model.py
```

**2. Package installation errors**
```bash
# Update pip and try again
pip install --upgrade pip
pip install -r requirements.txt
```

**3. Streamlit not starting**
```bash
# Check if Streamlit is installed
pip show streamlit
# Try running with explicit port
streamlit run app.py --server.port 8501
```

**4. Model training errors**
```bash
# Check dataset format
python -c "import pandas as pd; df = pd.read_csv('games.csv'); print(df.head())"
# Ensure required columns exist
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



## 👨‍💻 Author

**Rohit Pawar**
- GitHub: [@rohit3576](https://github.com/rohit3576)
- LinkedIn: [Rohit Pawar](https://www.linkedin.com/in/rohitpawar03576/)
- Email: rohit.pawar3576@example.com

## 🙏 Acknowledgments

- Steam for the game data
- Streamlit team for the amazing framework
- Scikit-learn for ML tools
- Plotly for visualization libraries


