# 🎬 Movie Success Prediction App (Streamlit Version)
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# =============================
# 📂 DATA LOADING & CLEANING
# =============================
@st.cache_data
def load_and_train_model():
    df = pd.read_csv('tmdb_5000_movies.csv')

    # Clean budget and revenue
    df['budget'] = df['budget'].replace(0, np.nan)
    df['revenue'] = df['revenue'].replace(0, np.nan)
    df['budget'] = df['budget'].fillna(df['budget'].mean())
    df['revenue'] = df['revenue'].fillna(df['revenue'].mean())

    # Create success ratio and label
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

# Load everything
df, model, acc, le, scaler, X = load_and_train_model()

# =============================
# 🎨 STREAMLIT UI
# =============================
st.set_page_config(page_title="🎬 Movie Success Predictor", layout="centered")
st.title("🎬 Movie Success Prediction App")
st.markdown("Predict whether a movie will be a **HIT 💥** or a **FLOP 💸** based on data from TMDB.")

# Sidebar info
st.sidebar.header("App Info")
st.sidebar.write(f"Model Accuracy: **{acc}%**")
st.sidebar.write("Built using Random Forest Classifier on TMDB 5000 Movies dataset.")
st.sidebar.write("Developed by **Bipin Balachandra Nayak** 🚀")

# User input section
st.subheader("🔍 Search for a Movie")
movie_name = st.text_input("Enter movie name:", placeholder="e.g., Avatar")

if st.button("Predict"):
    movie = df[df['title'].str.lower() == movie_name.lower()]
    if movie.empty:
        st.error(f"❌ Movie '{movie_name}' not found in dataset.")
    else:
        movie_data = movie[['budget','popularity','runtime','vote_average','vote_count','main_genre']]
        pred = model.predict(movie_data)
        pred_label = "💥 HIT" if pred[0]==1 else "💸 FLOP"
        st.success(f"🎯 Prediction for **{movie_name}** → **{pred_label}**")
        
        # Show movie details
        st.write("**Movie Details:**")
        st.dataframe(movie[['title','budget','popularity','runtime','vote_average','vote_count','main_genre']])

# =============================
# 📊 Feature Importance Chart
# =============================
st.subheader("📈 Feature Importance in Movie Success Prediction")

fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(x=model.feature_importances_, y=X.columns, ax=ax)
ax.set_title("Feature Importance in Random Forest")
ax.set_xlabel("Importance")
ax.set_ylabel("Feature")
st.pyplot(fig)

# =============================
# 🧠 Classification Report (Optional)
# =============================
with st.expander("Show Model Performance Details"):
    st.text("Random Forest Classification Report:")
    y_pred_all = model.predict(X)
    st.text(classification_report(df['success'], y_pred_all))
