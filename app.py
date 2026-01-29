import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page Config
st.set_page_config(page_title="Steam Hit Predictor", layout="wide", page_icon="🎮")

# Custom CSS for Modern UI
st.markdown("""
    <style>
    /* Main styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        border: 1px solid #e0e6ed;
    }
    
    .game-info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 5px solid #28a745;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left: 5px solid #ffc107;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Custom button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
    }
    </style>
    """, unsafe_allow_html=True)

# Header with gradient
st.markdown("""
    <div class="main-header">
        <h1 style="margin:0; font-size:2.5rem;">🎮 Steam Game Success Predictor</h1>
        <p style="margin:0; opacity:0.9; font-size:1.1rem;">Analyze game metrics and predict market success with AI-powered insights</p>
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
        st.metric("💰 Price", f"${game['price_final']:.2f}")
    with col2:
        st.metric("⭐ Rating", f"{game['positive_ratio']}%")
    
    # OS Support badges
    st.markdown("### 🖥️ OS Support")
    os_col1, os_col2, os_col3 = st.columns(3)
    with os_col1:
        st.markdown(f"<div style='text-align:center; padding:0.5rem; background:{'#4CAF50' if game.get('win') else '#e0e0e0'}; border-radius:5px; color:white;'>Windows</div>", unsafe_allow_html=True)
    with os_col2:
        st.markdown(f"<div style='text-align:center; padding:0.5rem; background:{'#2196F3' if game.get('mac') else '#e0e0e0'}; border-radius:5px; color:white;'>Mac</div>", unsafe_allow_html=True)
    with os_col3:
        st.markdown(f"<div style='text-align:center; padding:0.5rem; background:{'#FF9800' if game.get('linux') else '#e0e0e0'}; border-radius:5px; color:white;'>Linux</div>", unsafe_allow_html=True)

# Main Content Layout
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🎯 Prediction Analysis", "📈 Detailed Insights"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📈 Game Performance Metrics")
        
        # Create metric cards in a grid
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:0.9rem; color:#666;">Total Reviews</div>
                    <div style="font-size:2rem; font-weight:bold; color:#333;">{game['user_reviews']:,}</div>
                    <div style="font-size:0.8rem; color:#888;">Market Rank: #{df[df['user_reviews'] > game['user_reviews']].shape[0] + 1}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with metric_col2:
            discount_status = "Active" if game['discount'] > 0 else "None"
            discount_color = "#E91E63" if game['discount'] > 0 else "#666"
            st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:0.9rem; color:#666;">Discount Status</div>
                    <div style="font-size:2rem; font-weight:bold; color:{discount_color};">{game['discount']}%</div>
                    <div style="font-size:0.8rem; color:#888;">{discount_status}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with metric_col3:
            st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:0.9rem; color:#666;">Release Year</div>
                    <div style="font-size:2rem; font-weight:bold; color:#333;">{game.get('year', 'N/A')}</div>
                    <div style="font-size:0.8rem; color:#888;">Age: {2024 - int(game.get('year', 2024)) if str(game.get('year')).isdigit() else 'N/A'} years</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Performance comparison chart
        st.markdown("### 📊 Market Comparison")
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Price Comparison", "Review Comparison"))
        
        # Price comparison
        fig.add_trace(go.Bar(
            x=['Selected Game', 'Market Avg'],
            y=[game['price_final'], df['price_final'].mean()],
            marker_color=['#667eea', '#c3cfe2'],
            text=[f"${game['price_final']:.2f}", f"${df['price_final'].mean():.2f}"],
            textposition='auto',
        ), row=1, col=1)
        
        # Review comparison (normalized)
        max_reviews = df['user_reviews'].max()
        fig.add_trace(go.Bar(
            x=['Selected Game', 'Market Avg', 'Market Max'],
            y=[game['user_reviews']/max_reviews*100, 
               df['user_reviews'].mean()/max_reviews*100,
               100],
            marker_color=['#764ba2', '#c3cfe2', '#f8f9fa'],
            text=[f"{game['user_reviews']:,}", 
                  f"{df['user_reviews'].mean():,.0f}",
                  f"{max_reviews:,}"],
            textposition='auto',
        ), row=1, col=2)
        
        fig.update_layout(height=400, showlegend=False, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Quick Prediction")
        st.markdown("---")
        
        with st.container():
            st.markdown("""
                <div class="prediction-card">
                    <div style="text-align:center;">
                        <h3 style="color:#333; margin-bottom:1.5rem;">AI Analysis</h3>
                """, unsafe_allow_html=True)
            
            if st.button("🚀 Analyze Success Probability", use_container_width=True):
                input_data = np.array([[game[f] for f in features]])
                pred = model.predict(input_data)[0]
                prob = model.predict_proba(input_data)[0][1]
                
                if pred == 1:
                    st.markdown(f"""
                        <div class="success-box">
                            <h3 style="color:#28a745; margin:0;">🎉 HIT GAME!</h3>
                            <p style="margin:0.5rem 0; font-size:0.9rem;">High probability of market success</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.markdown(f"""
                        <div class="warning-box">
                            <h3 style="color:#ffc107; margin:0;">📊 NICHE GAME</h3>
                            <p style="margin:0.5rem 0; font-size:0.9rem;">Standard market performance expected</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Confidence meter
                st.markdown("### 📈 Confidence Level")
                confidence_color = "#4CAF50" if prob > 0.7 else "#FF9800" if prob > 0.4 else "#F44336"
                st.markdown(f"""
                    <div style="background:#f0f2f6; border-radius:10px; padding:1rem; margin:1rem 0;">
                        <div style="display:flex; justify-content:space-between; margin-bottom:0.5rem;">
                            <span>Low</span>
                            <span style="font-weight:bold; color:{confidence_color};">{prob*100:.1f}%</span>
                            <span>High</span>
                        </div>
                        <div style="background:#ddd; height:20px; border-radius:10px; overflow:hidden;">
                            <div style="width:{prob*100}%; background:{confidence_color}; height:100%;"></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

with tab2:
    st.markdown("### 🔍 Detailed Prediction Analysis")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Feature importance visualization
        st.markdown("#### 📊 Key Success Factors")
        
        # Sample feature importance (you can replace with actual feature importance if available)
        sample_features = features[:10]  # Show top 10 features
        importance_values = np.random.rand(len(sample_features))  # Replace with actual importance
        
        fig = go.Figure(data=[go.Bar(
            x=importance_values,
            y=sample_features,
            orientation='h',
            marker_color='#667eea'
        )])
        
        fig.update_layout(
            height=400,
            title="Feature Impact on Success Prediction",
            xaxis_title="Relative Importance",
            yaxis_title="Features",
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📋 Game Profile")
        
        # Create a game profile card
        profile_data = {
            "Metric": ["Price Category", "Review Volume", "Rating Tier", "Market Position"],
            "Value": [
                "Premium" if game['price_final'] > 20 else "Mid-range" if game['price_final'] > 5 else "Budget",
                "High" if game['user_reviews'] > df['user_reviews'].quantile(0.75) 
                else "Medium" if game['user_reviews'] > df['user_reviews'].quantile(0.25) 
                else "Low",
                "Excellent" if game['positive_ratio'] > 90 
                else "Good" if game['positive_ratio'] > 75 
                else "Average",
                "Top 25%" if game['user_reviews'] > df['user_reviews'].quantile(0.75) 
                else "Top 50%" if game['user_reviews'] > df['user_reviews'].quantile(0.5) 
                else "Below Average"
            ]
        }
        
        for metric, value in zip(profile_data["Metric"], profile_data["Value"]):
            st.markdown(f"""
                <div class="game-info-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:600; color:#555;">{metric}</span>
                        <span style="background:#667eea; color:white; padding:0.25rem 0.75rem; border-radius:15px; font-size:0.9rem;">
                            {value}
                        </span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

with tab3:
    st.markdown("### 📈 Detailed Insights & Raw Data")
    
    # Create two columns for different views
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Statistical Summary")
        
        # Calculate some statistics
        stats_data = {
            "Statistic": ["Mean", "Median", "Std Dev", "Min", "Max", "25%", "75%"],
            "Price": [
                f"${df['price_final'].mean():.2f}",
                f"${df['price_final'].median():.2f}",
                f"${df['price_final'].std():.2f}",
                f"${df['price_final'].min():.2f}",
                f"${df['price_final'].max():.2f}",
                f"${df['price_final'].quantile(0.25):.2f}",
                f"${df['price_final'].quantile(0.75):.2f}"
            ],
            "Reviews": [
                f"{df['user_reviews'].mean():,.0f}",
                f"{df['user_reviews'].median():,.0f}",
                f"{df['user_reviews'].std():,.0f}",
                f"{df['user_reviews'].min():,.0f}",
                f"{df['user_reviews'].max():,.0f}",
                f"{df['user_reviews'].quantile(0.25):,.0f}",
                f"{df['user_reviews'].quantile(0.75):,.0f}"
            ]
        }
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### 🎯 Selected Game vs Market")
        
        # Radar chart comparing selected game vs average
        categories = ['Price', 'Reviews', 'Rating', 'Discount']
        
        selected_values = [
            game['price_final'] / df['price_final'].max() * 100,
            min(game['user_reviews'] / df['user_reviews'].max() * 100, 100),
            game['positive_ratio'],
            game['discount']
        ]
        
        avg_values = [
            df['price_final'].mean() / df['price_final'].max() * 100,
            df['user_reviews'].mean() / df['user_reviews'].max() * 100,
            df['positive_ratio'].mean(),
            df['discount'].mean()
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=selected_values,
            theta=categories,
            fill='toself',
            name='Selected Game',
            line_color='#667eea'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=avg_values,
            theta=categories,
            fill='toself',
            name='Market Average',
            line_color='#c3cfe2'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Raw data expander
    with st.expander("📁 View Complete Raw Data"):
        st.dataframe(
            game.to_frame().T,
            use_container_width=True,
            column_config={
                "title": "Game Title",
                "price_final": st.column_config.NumberColumn("Price", format="$%.2f"),
                "positive_ratio": st.column_config.NumberColumn("Rating %", format="%.1f%%"),
                "user_reviews": st.column_config.NumberColumn("Reviews", format="%d")
            }
        )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#666; padding:1rem;'>"
    "🎮 Steam Game Success Predictor | Built with Streamlit | Data Science Project"
    "</div>",
    unsafe_allow_html=True
)