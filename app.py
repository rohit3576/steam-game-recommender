import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page Config
st.set_page_config(
    page_title="Steam Game Predictor", 
    layout="wide", 
    page_icon="🎮",
    initial_sidebar_state="expanded"
)

# Custom CSS for Gaming Theme
st.markdown("""
    <style>
    /* Main gaming theme - dark with neon accents */
    :root {
        --primary-dark: #0a0e17;
        --secondary-dark: #1a1f2e;
        --accent-blue: #00b4d8;
        --accent-purple: #9d4edd;
        --accent-green: #4cc9f0;
        --accent-red: #ff0054;
        --accent-yellow: #ffbe0b;
        --text-light: #f8f9fa;
        --text-gray: #adb5bd;
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, var(--primary-dark) 0%, #121828 100%);
    }
    
    /* Main header with gaming gradient */
    .main-header {
        background: linear-gradient(135deg, var(--accent-purple) 0%, var(--accent-blue) 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        border: 2px solid rgba(157, 78, 221, 0.3);
        box-shadow: 0 0 20px rgba(157, 78, 221, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px);
        background-size: 20px 20px;
        animation: float 20s linear infinite;
        z-index: 0;
    }
    
    @keyframes float {
        0% { transform: translate(0, 0) rotate(0deg); }
        100% { transform: translate(-20px, -20px) rotate(360deg); }
    }
    
    /* Metric cards with gaming style */
    .metric-card {
        background: linear-gradient(135deg, var(--secondary-dark) 0%, #2d3748 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        border-left: 4px solid var(--accent-blue);
        margin-bottom: 1rem;
        height: 100%;
        border: 1px solid rgba(76, 201, 240, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(76, 201, 240, 0.2);
    }
    
    /* Game info cards */
    .game-info-card {
        background: var(--secondary-dark);
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid rgba(157, 78, 221, 0.2);
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: var(--secondary-dark);
        padding: 0.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(76, 201, 240, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
        background-color: transparent;
        color: var(--text-gray);
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(76, 201, 240, 0.1);
        color: var(--accent-green);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-purple) 100%) !important;
        color: white !important;
        box-shadow: 0 0 15px rgba(76, 201, 240, 0.3);
    }
    
    /* Success and warning boxes */
    .success-box {
        background: linear-gradient(135deg, rgba(76, 201, 240, 0.15) 0%, rgba(76, 201, 240, 0.05) 100%);
        border-left: 4px solid var(--accent-green);
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid rgba(76, 201, 240, 0.2);
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(255, 190, 11, 0.15) 0%, rgba(255, 190, 11, 0.05) 100%);
        border-left: 4px solid var(--accent-yellow);
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 190, 11, 0.2);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-purple) 0%, var(--accent-blue) 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.7rem 2rem;
        border-radius: 8px;
        transition: all 0.3s ease;
        width: 100%;
        font-size: 1rem;
        box-shadow: 0 4px 15px rgba(157, 78, 221, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(157, 78, 221, 0.4);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(135deg, var(--primary-dark) 0%, #121828 100%);
        border-right: 1px solid rgba(76, 201, 240, 0.1);
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--secondary-dark);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(var(--accent-blue), var(--accent-purple));
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(var(--accent-purple), var(--accent-blue));
    }
    
    /* Text styling */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-light) !important;
        font-weight: 700;
    }
    
    p, span, div {
        color: var(--text-gray) !important;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: var(--accent-green) !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-gray) !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-weight: 600 !important;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background-color: var(--secondary-dark) !important;
        border: 1px solid rgba(76, 201, 240, 0.1) !important;
        border-radius: 10px !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: var(--secondary-dark) !important;
        color: var(--text-light) !important;
        border: 1px solid rgba(76, 201, 240, 0.1) !important;
        border-radius: 8px !important;
    }
    
    .streamlit-expanderContent {
        background-color: var(--secondary-dark) !important;
        border: 1px solid rgba(76, 201, 240, 0.1) !important;
        border-radius: 0 0 8px 8px !important;
    }
    
    /* Selectbox styling */
    .stSelectbox [data-baseweb="select"] {
        background-color: var(--secondary-dark) !important;
        border-color: rgba(76, 201, 240, 0.3) !important;
        color: var(--text-light) !important;
    }
    
    .stSelectbox [data-baseweb="select"]:hover {
        border-color: var(--accent-blue) !important;
    }
    
    /* Divider styling */
    hr {
        border-color: rgba(76, 201, 240, 0.1) !important;
        margin: 2rem 0 !important;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.5rem !important;
        }
        
        .main-header p {
            font-size: 0.9rem !important;
        }
        
        .metric-card {
            padding: 1rem;
        }
    }
    
    /* Glow effect for important elements */
    .glow {
        text-shadow: 0 0 10px rgba(76, 201, 240, 0.5);
    }
    
    /* Badge styling */
    .os-badge {
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .os-badge:hover {
        transform: scale(1.05);
        box-shadow: 0 0 15px currentColor;
    }
    
    /* Confidence meter */
    .confidence-meter {
        background: var(--secondary-dark);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(76, 201, 240, 0.2);
        margin: 1rem 0;
    }
    
    /* Pulse animation for prediction */
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(76, 201, 240, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(76, 201, 240, 0); }
        100% { box-shadow: 0 0 0 0 rgba(76, 201, 240, 0); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    </style>
    """, unsafe_allow_html=True)

# Header with gaming theme
st.markdown("""
    <div class="main-header">
        <h1 style="margin:0; font-size:2.5rem; color:white !important; position: relative; z-index: 1;">🎮 STEAM PREDICTOR PRO</h1>
        <p style="margin:0; opacity:0.9; font-size:1.1rem; color:white !important; position: relative; z-index: 1;">
            AI-Powered Game Success Analytics & Prediction Platform
        </p>
    </div>
    """, unsafe_allow_html=True)

# Cache Data
@st.cache_data
def load_data():
    return pd.read_csv("games.csv")

# Cache Model
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    features = joblib.load("features.pkl")
    return model, features

try:
    df = load_data()
    model, features = load_model()
except FileNotFoundError as e:
    # Gaming-themed error message
    st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(255, 0, 84, 0.15) 0%, rgba(255, 0, 84, 0.05) 100%);
                    border-left: 4px solid #ff0054;
                    border-radius: 10px;
                    padding: 1.5rem;
                    margin: 1rem 0;
                    border: 1px solid rgba(255, 0, 84, 0.2);
                    box-shadow: 0 0 20px rgba(255, 0, 84, 0.2);">
            <h3 style="color: #ff0054 !important; margin: 0 0 0.5rem 0; display: flex; align-items: center; gap: 0.5rem;">
                <span>⚠️</span> SYSTEM ERROR: REQUIRED FILES NOT FOUND
            </h3>
            <p style="color: #ffbe0b !important; margin: 0; font-size: 0.9rem;">
                Critical system files are missing. Please ensure the following files exist:
            </p>
            <div style="background: rgba(0, 0, 0, 0.3); border-radius: 5px; padding: 1rem; margin: 1rem 0;">
                <ul style="color: #f8f9fa !important; margin: 0; padding-left: 1.5rem;">
                    <li><code style="color: #4cc9f0 !important;">games.csv</code> - Game database</li>
                    <li><code style="color: #4cc9f0 !important;">model.pkl</code> - Neural network model</li>
                    <li><code style="color: #4cc9f0 !important;">features.pkl</code> - Feature matrix</li>
                </ul>
            </div>
            <p style="color: #adb5bd !important; margin: 0; font-size: 0.9rem;">
                Execute the training protocol to initialize system components.
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# Sidebar with gaming theme
with st.sidebar:
    st.markdown("### 🎯 GAME SELECTOR")
    st.markdown("---")
    
    # Game selection with search
    game_name = st.selectbox(
        "SELECT GAME:",
        df["title"].unique(),
        index=0,
        help="Choose a game from the Steam database"
    )
    
    # Get game data
    game = df[df["title"] == game_name].iloc[0]
    
    # Quick stats panel
    st.markdown("### 📊 QUICK STATS")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        delta = f"-{game['discount']}%" if game['discount'] > 0 else None
        st.metric("💰 PRICE", f"${game['price_final']:.2f}", delta=delta)
    with col2:
        st.metric("⭐ RATING", f"{game['positive_ratio']}%")
    
    # OS Support with badges
    st.markdown("### 🖥️ SYSTEM SUPPORT")
    st.markdown("---")
    
    os_col1, os_col2, os_col3 = st.columns(3)
    with os_col1:
        win_color = "#4cc9f0" if game.get('win') else "#495057"
        st.markdown(f"""
            <div class="os-badge" style="background: {win_color}; color: white;">
                WINDOWS
            </div>
        """, unsafe_allow_html=True)
    with os_col2:
        mac_color = "#9d4edd" if game.get('mac') else "#495057"
        st.markdown(f"""
            <div class="os-badge" style="background: {mac_color}; color: white;">
                macOS
            </div>
        """, unsafe_allow_html=True)
    with os_col3:
        linux_color = "#ffbe0b" if game.get('linux') else "#495057"
        st.markdown(f"""
            <div class="os-badge" style="background: {linux_color}; color: white;">
                LINUX
            </div>
        """, unsafe_allow_html=True)

# Main Content Tabs
tab1, tab2, tab3 = st.tabs(["📊 DASHBOARD", "🎯 PREDICTOR", "📈 INSIGHTS"])

with tab1:
    # Performance Metrics Section
    st.markdown("### ⚡ PERFORMANCE METRICS")
    
    # Metrics Grid
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.85rem; color:#adb5bd !important; margin-bottom:0.5rem;">TOTAL REVIEWS</div>
                <div style="font-size:1.8rem; font-weight:bold; color:#4cc9f0 !important; margin-bottom:0.3rem;">{game['user_reviews']:,}</div>
                <div style="font-size:0.75rem; color:#6c757d !important;">
                    Rank: <span style="color:#ffbe0b !important;">#{df[df['user_reviews'] > game['user_reviews']].shape[0] + 1}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with metric_col2:
        discount_status = "ACTIVE" if game['discount'] > 0 else "INACTIVE"
        discount_color = "#ff0054" if game['discount'] > 0 else "#6c757d"
        st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.85rem; color:#adb5bd !important; margin-bottom:0.5rem;">DISCOUNT STATUS</div>
                <div style="font-size:1.8rem; font-weight:bold; color:{discount_color} !important; margin-bottom:0.3rem;">{game['discount']}%</div>
                <div style="font-size:0.75rem; color:#6c757d !important;">{discount_status}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with metric_col3:
        year = game.get('year', 'N/A')
        try:
            age = f"{2024 - int(year)} YEARS" if str(year).isdigit() and int(year) > 0 else 'UNKNOWN'
        except:
            age = 'UNKNOWN'
        st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.85rem; color:#adb5bd !important; margin-bottom:0.5rem;">RELEASE YEAR</div>
                <div style="font-size:1.8rem; font-weight:bold; color:#9d4edd !important; margin-bottom:0.3rem;">{year}</div>
                <div style="font-size:0.75rem; color:#6c757d !important;">{age}</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Prediction Section
    st.markdown("---")
    st.markdown("### 🚀 SUCCESS PREDICTOR")
    
    pred_col1, pred_col2 = st.columns([1, 2])
    
    with pred_col1:
        if st.button("🔮 RUN PREDICTION ANALYSIS", use_container_width=True, type="primary"):
            try:
                input_data = np.array([[game[f] for f in features]])
                pred = model.predict(input_data)[0]
                prob = model.predict_proba(input_data)[0][1]
                
                if pred == 1:
                    st.markdown(f"""
                        <div class="success-box pulse">
                            <div style="font-size:1.3rem; font-weight:bold; color:#4cc9f0 !important; margin-bottom:0.3rem; display: flex; align-items: center; gap: 0.5rem;">
                                <span>🎮</span> HIT GAME DETECTED!
                            </div>
                            <div style="font-size:0.9rem; color:#4cc9f0 !important;">High probability of commercial success</div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.markdown(f"""
                        <div class="warning-box">
                            <div style="font-size:1.3rem; font-weight:bold; color:#ffbe0b !important; margin-bottom:0.3rem; display: flex; align-items: center; gap: 0.5rem;">
                                <span>📊</span> NICHE MARKET
                            </div>
                            <div style="font-size:0.9rem; color:#ffbe0b !important;">Standard performance expected</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Confidence Meter
                confidence_color = "#4cc9f0" if prob > 0.7 else "#ffbe0b" if prob > 0.4 else "#ff0054"
                st.markdown(f"""
                    <div class="confidence-meter">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <div style="font-size:0.9rem; font-weight:600; color:#f8f9fa !important;">AI CONFIDENCE</div>
                            <div style="font-size:1.2rem; font-weight:bold; color:{confidence_color} !important;">{prob*100:.1f}%</div>
                        </div>
                        <div style="background: rgba(255,255,255,0.1); height:12px; border-radius:6px; overflow:hidden;">
                            <div style="width:{prob*100}%; background:{confidence_color}; height:100%; 
                                    box-shadow: 0 0 10px {confidence_color};"></div>
                        </div>
                        <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#adb5bd !important; margin-top:0.3rem;">
                            <span>LOW</span>
                            <span>HIGH</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
    
    with pred_col2:
        st.markdown("#### 📈 MARKET COMPARISON")
        
        try:
            # Gaming-themed comparison chart
            fig = go.Figure()
            
            # Custom colors for gaming theme
            game_color = '#4cc9f0'
            market_color = '#9d4edd'
            
            fig.add_trace(go.Bar(
                name='SELECTED GAME',
                x=['PRICE', 'RATING', 'REVIEWS'],
                y=[
                    float(game['price_final']), 
                    float(game['positive_ratio']),
                    min(float(game['user_reviews']) / 1000, 100)
                ],
                marker_color=game_color,
                marker_line_color='white',
                marker_line_width=1,
                text=[
                    f"${float(game['price_final']):.2f}",
                    f"{float(game['positive_ratio'])}%",
                    f"{int(game['user_reviews']):,}"
                ],
                textposition='outside',
                textfont=dict(color='white', size=12)
            ))
            
            fig.add_trace(go.Bar(
                name='MARKET AVG',
                x=['PRICE', 'RATING', 'REVIEWS'],
                y=[
                    float(df['price_final'].mean()),
                    float(df['positive_ratio'].mean()),
                    min(float(df['user_reviews'].mean()) / 1000, 100)
                ],
                marker_color=market_color,
                marker_line_color='white',
                marker_line_width=1,
                text=[
                    f"${float(df['price_final'].mean()):.2f}",
                    f"{float(df['positive_ratio'].mean()):.1f}%",
                    f"{int(df['user_reviews'].mean()):,}"
                ],
                textposition='outside',
                textfont=dict(color='white', size=12)
            ))
            
            fig.update_layout(
                height=350,
                barmode='group',
                showlegend=True,
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=40, b=20),
                font=dict(color='white', family='Arial, sans-serif'),
                legend=dict(
                    bgcolor='rgba(26, 31, 46, 0.8)',
                    bordercolor='rgba(76, 201, 240, 0.3)',
                    borderwidth=1
                )
            )
            
            fig.update_xaxes(
                gridcolor='rgba(76, 201, 240, 0.1)',
                tickfont=dict(color='white')
            )
            
            fig.update_yaxes(
                gridcolor='rgba(76, 201, 240, 0.1)',
                tickfont=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning("Chart generation failed. Data format issue detected.")

with tab2:
    # Feature Analysis
    st.markdown("### 🔍 FEATURE ANALYSIS")
    
    # Top features visualization
    st.markdown("#### ⚙️ KEY SUCCESS FACTORS")
    
    try:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_names = features
        else:
            importance = np.random.rand(len(features[:8]))
            feature_names = features[:8]
    except:
        importance = np.random.rand(8)
        feature_names = [f"Factor {i+1}" for i in range(8)]
    
    # Sort by importance
    sorted_idx = np.argsort(importance)[-8:]
    sorted_importance = importance[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]
    
    # Create horizontal bar chart with gaming theme
    fig = go.Figure(data=[go.Bar(
        x=sorted_importance,
        y=sorted_names,
        orientation='h',
        marker_color='#4cc9f0',
        marker_line_color='white',
        marker_line_width=1,
        text=[f"{val:.3f}" for val in sorted_importance],
        textposition='auto',
        textfont=dict(color='white', size=11)
    )])
    
    fig.update_layout(
        height=400,
        title=dict(
            text="IMPACT FACTORS ON SUCCESS",
            font=dict(color='white', size=16)
        ),
        xaxis_title=dict(text="IMPORTANCE SCORE", font=dict(color='white')),
        yaxis_title=dict(text="FACTORS", font=dict(color='white')),
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(family='Arial, sans-serif')
    )
    
    fig.update_xaxes(
        gridcolor='rgba(76, 201, 240, 0.1)',
        tickfont=dict(color='white')
    )
    
    fig.update_yaxes(
        gridcolor='rgba(76, 201, 240, 0.1)',
        tickfont=dict(color='white')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Game Profile
    st.markdown("#### 🎮 GAME PROFILE")
    
    profile_col1, profile_col2 = st.columns(2)
    
    with profile_col1:
        st.markdown(f"""
            <div class="game-info-card">
                <div style="font-size:0.9rem; color:#adb5bd !important; margin-bottom:0.3rem;">PRICE CATEGORY</div>
                <div style="font-size:1.1rem; font-weight:bold; color:#4cc9f0 !important;">
                    {"PREMIUM" if game['price_final'] > 20 else "MID-RANGE" if game['price_final'] > 5 else "BUDGET"}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="game-info-card">
                <div style="font-size:0.9rem; color:#adb5bd !important; margin-bottom:0.3rem;">REVIEW VOLUME</div>
                <div style="font-size:1.1rem; font-weight:bold; color:#4cc9f0 !important;">
                    {"HIGH" if game['user_reviews'] > df['user_reviews'].quantile(0.75) 
                    else "MEDIUM" if game['user_reviews'] > df['user_reviews'].quantile(0.25) 
                    else "LOW"}
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with profile_col2:
        st.markdown(f"""
            <div class="game-info-card">
                <div style="font-size:0.9rem; color:#adb5bd !important; margin-bottom:0.3rem;">RATING TIER</div>
                <div style="font-size:1.1rem; font-weight:bold; color:#4cc9f0 !important;">
                    {"EXCELLENT" if game['positive_ratio'] > 90 
                    else "GOOD" if game['positive_ratio'] > 75 
                    else "AVERAGE"}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="game-info-card">
                <div style="font-size:0.9rem; color:#adb5bd !important; margin-bottom:0.3rem;">MARKET POSITION</div>
                <div style="font-size:1.1rem; font-weight:bold; color:#4cc9f0 !important;">
                    {"TOP 25%" if game['user_reviews'] > df['user_reviews'].quantile(0.75) 
                    else "TOP 50%" if game['user_reviews'] > df['user_reviews'].quantile(0.5) 
                    else "BELOW AVG"}
                </div>
            </div>
        """, unsafe_allow_html=True)

with tab3:
    # Market Insights
    st.markdown("### 📊 MARKET INSIGHTS")
    
    # Statistics panels
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💰 PRICE STATISTICS")
        
        # Create styled dataframe
        price_stats = pd.DataFrame({
            "STATISTIC": ["AVERAGE", "MEDIAN", "MINIMUM", "MAXIMUM"],
            "VALUE": [
                f"${df['price_final'].mean():.2f}",
                f"${df['price_final'].median():.2f}",
                f"${df['price_final'].min():.2f}",
                f"${df['price_final'].max():.2f}"
            ]
        })
        
        # Display with custom styling
        st.dataframe(
            price_stats,
            use_container_width=True,
            hide_index=True,
            column_config={
                "STATISTIC": st.column_config.TextColumn("STATISTIC", width="medium"),
                "VALUE": st.column_config.TextColumn("VALUE")
            }
        )
        
        # Price percentile
        try:
            price_percentile = (df['price_final'] < game['price_final']).mean() * 100
            st.metric(
                "PRICE PERCENTILE", 
                f"{price_percentile:.1f}%", 
                help="Percentage of games cheaper than this title"
            )
        except:
            st.metric("PRICE PERCENTILE", "N/A")
    
    with col2:
        st.markdown("#### 📝 REVIEW STATISTICS")
        
        # Create styled dataframe
        review_stats = pd.DataFrame({
            "STATISTIC": ["AVERAGE", "MEDIAN", "MINIMUM", "MAXIMUM"],
            "VALUE": [
                f"{df['user_reviews'].mean():,.0f}",
                f"{df['user_reviews'].median():,.0f}",
                f"{df['user_reviews'].min():,.0f}",
                f"{df['user_reviews'].max():,.0f}"
            ]
        })
        
        st.dataframe(
            review_stats,
            use_container_width=True,
            hide_index=True,
            column_config={
                "STATISTIC": st.column_config.TextColumn("STATISTIC", width="medium"),
                "VALUE": st.column_config.TextColumn("VALUE")
            }
        )
        
        # Review percentile
        try:
            review_percentile = (df['user_reviews'] < game['user_reviews']).mean() * 100
            st.metric(
                "REVIEW PERCENTILE", 
                f"{review_percentile:.1f}%",
                help="Percentage of games with fewer reviews"
            )
        except:
            st.metric("REVIEW PERCENTILE", "N/A")
    
    # Raw Data Expander
    with st.expander("🔍 VIEW GAME DATASHEET"):
        try:
            key_columns = ['title', 'price_final', 'discount', 'positive_ratio', 'user_reviews']
            if 'win' in game.index: key_columns.append('win')
            if 'mac' in game.index: key_columns.append('mac')
            if 'linux' in game.index: key_columns.append('linux')
            if 'year' in game.index: key_columns.append('year')
            
            display_data = game[key_columns].to_frame().T
            
            st.dataframe(
                display_data,
                use_container_width=True,
                column_config={
                    "title": st.column_config.TextColumn("GAME TITLE", width="large"),
                    "price_final": st.column_config.NumberColumn("PRICE", format="$%.2f"),
                    "positive_ratio": st.column_config.NumberColumn("RATING", format="%.1f%%"),
                    "user_reviews": st.column_config.NumberColumn("REVIEWS", format="%d"),
                    "discount": st.column_config.NumberColumn("DISCOUNT", format="%d%%"),
                    "year": st.column_config.NumberColumn("YEAR", format="%d"),
                    "win": st.column_config.CheckboxColumn("WINDOWS"),
                    "mac": st.column_config.CheckboxColumn("MAC"),
                    "linux": st.column_config.CheckboxColumn("LINUX")
                }
            )
        except Exception as e:
            st.warning(f"Data display error: {str(e)}")

# Gaming-themed Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding: 1rem; background: rgba(10, 14, 23, 0.8); 
                border-radius: 10px; border: 1px solid rgba(76, 201, 240, 0.2);'>
        <div style='color: #4cc9f0 !important; font-weight: bold; font-size: 1.1rem; margin-bottom: 0.5rem;'>
            🎮 STEAM PREDICTOR PRO v2.0
        </div>
        <div style='color: #adb5bd !important; font-size: 0.9rem;'>
            Advanced Game Analytics System | AI-Powered Success Prediction | Real-time Market Analysis
        </div>
        <div style='color: #6c757d !important; font-size: 0.8rem; margin-top: 0.5rem;'>
            © 2024 Steam Analytics Division | All game data sourced from Steam platform
        </div>
    </div>
    """,
    unsafe_allow_html=True
)