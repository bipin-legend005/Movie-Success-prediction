# 🎬 Movie Success Prediction App (2025 Edition)
import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# =============================
# 📂 DATA LOADING & MODEL TRAINING
# =============================
@st.cache_data
def load_and_train_model():
    df = pd.read_csv('tmdb_5000_movies.csv')

    # Clean budget and revenue
    df['budget'] = df['budget'].replace(0, np.nan)
    df['revenue'] = df['revenue'].replace(0, np.nan)
    df['budget'] = df['budget'].fillna(df['budget'].mean())
    df['revenue'] = df['revenue'].fillna(df['revenue'].mean())

    # Success ratio and label
    df['success_ratio'] = df['revenue'] / df['budget']
    df['success'] = df['success_ratio'].apply(lambda x: 1 if x > 1.5 else 0)

    # Extract main genre
    def extract_main_genre(x):
        try:
            genres = eval(x)
            if isinstance(genres, list) and len(genres) > 0:
                return genres[0]['name']
        except:
            return 'Unknown'
        return 'Unknown'

    df['main_genre'] = df['genres'].apply(extract_main_genre)

    # Select relevant features
    features = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count', 'main_genre']
    df = df[features + ['title', 'success']].dropna()

    # Encode genre
    le = LabelEncoder()
    df['main_genre'] = le.fit_transform(df['main_genre'])

    # Scale numeric columns
    scaler = StandardScaler()
    df[['budget','popularity','runtime','vote_average','vote_count']] = scaler.fit_transform(
        df[['budget','popularity','runtime','vote_average','vote_count']]
    )

    X = df.drop(['success', 'title'], axis=1)
    y = df['success']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = round(accuracy_score(y_test, y_pred)*100, 2)

    return df, model, acc, le, scaler, X

df, model, acc, le, scaler, X = load_and_train_model()

# =============================
# 🎨 STREAMLIT UI (2K25 Edition)
# =============================
st.set_page_config(page_title="🎬 Movie Success Predictor 2K25", layout="centered")

st.markdown(
    "<h1 style='text-align:center; color:#00ADB5;'>🎬 Movie Success Predictor (2025 Edition)</h1>",
    unsafe_allow_html=True,
)
st.caption("AI-powered prediction of movie success using TMDB dataset.")

# Sidebar Info
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/7429/7429750.png", width=100)
st.sidebar.title("ℹ️ App Info")
st.sidebar.write(f"**Model Accuracy:** `{acc}%`")
st.sidebar.write("Algorithm: **Random Forest Classifier**")
st.sidebar.write("Developer: **Bipin Balachandra Nayak 🚀**")
st.sidebar.write("Data Source: TMDB 5000 Movies Dataset")

# =============================
# 🎬 Movie Dropdown
# =============================
st.subheader("🔍 Choose a Movie to Predict")

# Dropdown with search bar
movie_list = sorted(df['title'].unique().tolist())
selected_movie = st.selectbox("🎞️ Select or Search Movie", movie_list, index=0, key="movie_dropdown")

# Optional: Fetch poster from TMDB API
def fetch_poster(movie_name):
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key=8b2b2caa927ee8bcb99591b1b0e7f2dd&query={movie_name}"
        data = requests.get(url).json()
        if data['results']:
            poster_path = data['results'][0]['poster_path']
            full_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
            return full_path
    except:
        return None
    return None

# Prediction
if st.button("🚀 Predict Success"):
    movie = df[df['title'] == selected_movie]
    if movie.empty:
        st.error("❌ Movie not found in dataset.")
    else:
        movie_data = movie[['budget','popularity','runtime','vote_average','vote_count','main_genre']]
        pred = model.predict(movie_data)
        pred_label = "💥 HIT" if pred[0]==1 else "💸 FLOP"
        st.success(f"🎯 **Prediction:** {selected_movie} → {pred_label}")

        # Display poster
        poster_url = fetch_poster(selected_movie)
        if poster_url:
            st.image(poster_url, caption=selected_movie, width=250)
        else:
            st.warning("Poster not found, but prediction completed ✅")

        # Movie details
        with st.expander("🎬 View Movie Details"):
            st.dataframe(movie[['title','budget','popularity','runtime','vote_average','vote_count','main_genre']])

# =============================
# 📊 Feature Importance
# =============================
st.markdown("### 📈 Feature Importance in Success Prediction")
fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(x=model.feature_importances_, y=X.columns, ax=ax, palette="coolwarm")
ax.set_title("🎬 Random Forest Feature Importance", fontsize=12)
ax.set_xlabel("Importance")
ax.set_ylabel("Feature")
st.pyplot(fig)

# =============================
# 🧠 Model Performance
# =============================
with st.expander("🔍 Show Model Performance Report"):
    st.text("Random Forest Classification Report:")
    y_pred_all = model.predict(X)
    st.text(classification_report(df['success'], y_pred_all))

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>🔥 Built by <b>Bipin Balachandra Nayak</b> | Powered by Streamlit 2K25</p>",
    unsafe_allow_html=True,
)
