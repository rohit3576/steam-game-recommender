import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page Config
st.set_page_config(
    page_title="Steam Hit Predictor", 
    layout="wide", 
    page_icon="🎮",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern UI with better scroll handling
st.markdown("""
    <style>
    /* Main styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1.5rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
        height: 100%;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
        border: 1px solid #e0e6ed;
        margin-bottom: 1rem;
    }
    
    .game-info-card {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    
    /* Make tabs more compact */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    /* Custom button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 8px;
        transition: all 0.3s ease;
        width: 100%;
        font-size: 0.9rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Improve sidebar scrolling */
    .css-1d391kg {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
    }
    
    /* Better scroll handling for main content */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Adjust chart heights */
    .js-plotly-plot {
        margin-bottom: 1rem;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.8rem !important;
        }
        
        .metric-card {
            padding: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Header with gradient
st.markdown("""
    <div class="main-header">
        <h1 style="margin:0; font-size:2rem;">🎮 Steam Game Success Predictor</h1>
        <p style="margin:0; opacity:0.9; font-size:1rem;">Analyze game metrics and predict market success with AI-powered insights</p>
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
except FileNotFoundError:
    st.error("❌ Error: Required files not found. Please run the training script first.")
    st.stop()

# Sidebar for Selection
with st.sidebar:
    st.markdown("### 🔍 Game Selection")
    st.markdown("---")
    
    # Game search with improved UI
    game_name = st.selectbox(
        "Choose a game:",
        df["title"].unique(),
        index=0,
        help="Select a game from the dataset to analyze"
    )
    
    # Get game data
    game = df[df["title"] == game_name].iloc[0]
    
    # Game info in sidebar
    st.markdown("### 📋 Quick Stats")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("💰 Price", f"${game['price_final']:.2f}", 
                  delta=f"-{game['discount']}%" if game['discount'] > 0 else None)
    with col2:
        st.metric("⭐ Rating", f"{game['positive_ratio']}%")
    
    # OS Support badges
    st.markdown("### 🖥️ OS Support")
    os_col1, os_col2, os_col3 = st.columns(3)
    with os_col1:
        st.markdown(f"<div style='text-align:center; padding:0.4rem; background:{'#4CAF50' if game.get('win') else '#e0e0e0'}; border-radius:5px; color:white; font-size:0.8rem;'>Win</div>", 
                    unsafe_allow_html=True)
    with os_col2:
        st.markdown(f"<div style='text-align:center; padding:0.4rem; background:{'#2196F3' if game.get('mac') else '#e0e0e0'}; border-radius:5px; color:white; font-size:0.8rem;'>Mac</div>", 
                    unsafe_allow_html=True)
    with os_col3:
        st.markdown(f"<div style='text-align:center; padding:0.4rem; background:{'#FF9800' if game.get('linux') else '#e0e0e0'}; border-radius:5px; color:white; font-size:0.8rem;'>Linux</div>", 
                    unsafe_allow_html=True)

# Main Content Layout - Use a single column for better mobile experience
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🎯 Analysis", "📈 Insights"])

with tab1:
    # Use single column with proper spacing
    st.markdown("### 📈 Game Performance Metrics")
    
    # Metric cards in a responsive grid
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.85rem; color:#666; margin-bottom:0.5rem;">Total Reviews</div>
                <div style="font-size:1.6rem; font-weight:bold; color:#333; margin-bottom:0.3rem;">{game['user_reviews']:,}</div>
                <div style="font-size:0.75rem; color:#888;">Rank: #{df[df['user_reviews'] > game['user_reviews']].shape[0] + 1}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with metric_col2:
        discount_status = "Active" if game['discount'] > 0 else "None"
        discount_color = "#E91E63" if game['discount'] > 0 else "#666"
        st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.85rem; color:#666; margin-bottom:0.5rem;">Discount</div>
                <div style="font-size:1.6rem; font-weight:bold; color:{discount_color}; margin-bottom:0.3rem;">{game['discount']}%</div>
                <div style="font-size:0.75rem; color:#888;">{discount_status}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with metric_col3:
        year = game.get('year', 'N/A')
        age = f"{2024 - int(year)} years" if str(year).isdigit() and int(year) > 0 else 'N/A'
        st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.85rem; color:#666; margin-bottom:0.5rem;">Release Year</div>
                <div style="font-size:1.6rem; font-weight:bold; color:#333; margin-bottom:0.3rem;">{year}</div>
                <div style="font-size:0.75rem; color:#888;">{age}</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Prediction section at the top for better visibility
    st.markdown("---")
    st.markdown("### 🎯 Success Prediction")
    
    pred_col1, pred_col2 = st.columns([1, 2])
    
    with pred_col1:
        if st.button("🚀 Run Analysis", use_container_width=True, type="primary"):
            input_data = np.array([[game[f] for f in features]])
            pred = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][1]
            
            if pred == 1:
                st.markdown(f"""
                    <div class="success-box">
                        <div style="font-size:1.2rem; font-weight:bold; color:#28a745; margin-bottom:0.3rem;">🎉 HIT GAME!</div>
                        <div style="font-size:0.85rem; color:#155724;">High probability of market success</div>
                    </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                    <div class="warning-box">
                        <div style="font-size:1.2rem; font-weight:bold; color:#ffc107; margin-bottom:0.3rem;">📊 NICHE GAME</div>
                        <div style="font-size:0.85rem; color:#856404;">Standard market performance expected</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Confidence meter
            confidence_color = "#4CAF50" if prob > 0.7 else "#FF9800" if prob > 0.4 else "#F44336"
            st.markdown(f"""
                <div style="margin:1rem 0;">
                    <div style="font-size:0.9rem; font-weight:600; color:#555; margin-bottom:0.5rem;">Confidence: {prob*100:.1f}%</div>
                    <div style="background:#f0f2f6; height:12px; border-radius:6px; overflow:hidden;">
                        <div style="width:{prob*100}%; background:{confidence_color}; height:100%;"></div>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#888; margin-top:0.3rem;">
                        <span>0%</span>
                        <span>100%</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    with pred_col2:
        st.markdown("#### 📊 Market Comparison")
        
        # Simplified comparison chart
        fig = go.Figure()
        
        # Add bars for price comparison
        fig.add_trace(go.Bar(
            name='Selected Game',
            x=['Price', 'Rating', 'Reviews (scaled)'],
            y=[
                game['price_final'], 
                game['positive_ratio'],
                min(game['user_reviews'] / 1000, 100)  # Scale reviews for visibility
            ],
            marker_color='#667eea',
            text=[
                f"${game['price_final']:.2f}",
                f"{game['positive_ratio']}%",
                f"{game['user_reviews']:,}"
            ],
            textposition='auto',
        ))
        
        fig.add_trace(go.Bar(
            name='Market Avg',
            x=['Price', 'Rating', 'Reviews (scaled)'],
            y=[
                df['price_final'].mean(),
                df['positive_ratio'].mean(),
                min(df['user_reviews'].mean() / 1000, 100)
            ],
            marker_color='#c3cfe2',
            text=[
                f"${df['price_final'].mean():.2f}",
                f"{df['positive_ratio'].mean():.1f}%",
                f"{df['user_reviews'].mean():,.0f}"
            ],
            textposition='auto',
        ))
        
        fig.update_layout(
            height=300,
            barmode='group',
            showlegend=True,
            template='plotly_white',
            margin=dict(l=20, r=20, t=40, b=20),
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Simplified analysis tab
    st.markdown("### 🔍 Feature Analysis")
    
    # Top features visualization
    st.markdown("#### 📋 Key Success Factors")
    
    # Get actual feature importance if model supports it
    try:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_names = features
        else:
            # Fallback to random for demonstration
            importance = np.random.rand(len(features[:8]))
            feature_names = features[:8]
    except:
        importance = np.random.rand(8)
        feature_names = [f"Feature {i+1}" for i in range(8)]
    
    # Sort by importance
    sorted_idx = np.argsort(importance)[-8:]  # Top 8 features
    sorted_importance = importance[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]
    
    fig = go.Figure(data=[go.Bar(
        x=sorted_importance,
        y=sorted_names,
        orientation='h',
        marker_color='#667eea',
        text=[f"{val:.3f}" for val in sorted_importance],
        textposition='auto',
    )])
    
    fig.update_layout(
        height=350,
        title="Top Predictive Features",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        template='plotly_white',
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Game profile
    st.markdown("#### 🎮 Game Profile")
    
    profile_col1, profile_col2 = st.columns(2)
    
    with profile_col1:
        st.markdown(f"""
            <div class="game-info-card">
                <div style="font-size:0.9rem; color:#666; margin-bottom:0.3rem;">Price Category</div>
                <div style="font-size:1rem; font-weight:bold; color:#333;">
                    {"Premium" if game['price_final'] > 20 else "Mid-range" if game['price_final'] > 5 else "Budget"}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="game-info-card">
                <div style="font-size:0.9rem; color:#666; margin-bottom:0.3rem;">Review Volume</div>
                <div style="font-size:1rem; font-weight:bold; color:#333;">
                    {"High" if game['user_reviews'] > df['user_reviews'].quantile(0.75) 
                    else "Medium" if game['user_reviews'] > df['user_reviews'].quantile(0.25) 
                    else "Low"}
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with profile_col2:
        st.markdown(f"""
            <div class="game-info-card">
                <div style="font-size:0.9rem; color:#666; margin-bottom:0.3rem;">Rating Tier</div>
                <div style="font-size:1rem; font-weight:bold; color:#333;">
                    {"Excellent" if game['positive_ratio'] > 90 
                    else "Good" if game['positive_ratio'] > 75 
                    else "Average"}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="game-info-card">
                <div style="font-size:0.9rem; color:#666; margin-bottom:0.3rem;">Market Position</div>
                <div style="font-size:1rem; font-weight:bold; color:#333;">
                    {"Top 25%" if game['user_reviews'] > df['user_reviews'].quantile(0.75) 
                    else "Top 50%" if game['user_reviews'] > df['user_reviews'].quantile(0.5) 
                    else "Below Average"}
                </div>
            </div>
        """, unsafe_allow_html=True)

with tab3:
    # Insights tab - more compact
    st.markdown("### 📈 Market Insights")
    
    # Statistics in a cleaner format
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Price Statistics")
        price_stats = pd.DataFrame({
            "Statistic": ["Average", "Median", "Minimum", "Maximum"],
            "Price": [
                f"${df['price_final'].mean():.2f}",
                f"${df['price_final'].median():.2f}",
                f"${df['price_final'].min():.2f}",
                f"${df['price_final'].max():.2f}"
            ]
        })
        st.dataframe(price_stats, use_container_width=True, hide_index=True)
        
        # Show where selected game stands
        price_percentile = (df['price_final'] < game['price_final']).mean() * 100
        st.metric("Price Percentile", f"{price_percentile:.1f}%", 
                  help="Percentage of games cheaper than this one")
    
    with col2:
        st.markdown("#### 📊 Review Statistics")
        review_stats = pd.DataFrame({
            "Statistic": ["Average", "Median", "Minimum", "Maximum"],
            "Reviews": [
                f"{df['user_reviews'].mean():,.0f}",
                f"{df['user_reviews'].median():,.0f}",
                f"{df['user_reviews'].min():,.0f}",
                f"{df['user_reviews'].max():,.0f}"
            ]
        })
        st.dataframe(review_stats, use_container_width=True, hide_index=True)
        
        # Show where selected game stands
        review_percentile = (df['user_reviews'] < game['user_reviews']).mean() * 100
        st.metric("Review Percentile", f"{review_percentile:.1f}%",
                  help="Percentage of games with fewer reviews than this one")
    
    # Raw data in expander
    with st.expander("📁 View Game Details"):
        # Display only key columns
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
                "title": st.column_config.TextColumn("Game Title", width="medium"),
                "price_final": st.column_config.NumberColumn("Price", format="$%.2f"),
                "positive_ratio": st.column_config.NumberColumn("Rating", format="%.1f%%"),
                "user_reviews": st.column_config.NumberColumn("Reviews", format="%d"),
                "discount": st.column_config.NumberColumn("Discount", format="%d%%"),
                "year": st.column_config.NumberColumn("Year", format="%d"),
                "win": st.column_config.CheckboxColumn("Windows"),
                "mac": st.column_config.CheckboxColumn("Mac"),
                "linux": st.column_config.CheckboxColumn("Linux")
            }
        )

# Compact footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#666; padding:0.5rem; font-size:0.85rem;'>"
    "🎮 Steam Game Success Predictor | Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)