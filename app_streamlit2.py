import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from joblib import load

# Load the trained model
model = load('trained_rf_model.joblib')
scaler = load('scaler_model.joblib')

st.title('Player Ratings Predictor using RandomForest')

# Take user input
st.sidebar.header('Input Player Features')

def user_input():
    reactions = st.sidebar.slider('Movement Reactions', 30, 100, 60)
    composure = st.sidebar.slider('Mentality Composure', 20, 100, 50)
    passing = st.sidebar.slider('Passing', 10, 100, 50)
    potential = st.sidebar.slider('Potential', 40, 100, 70)
    release_clause_eur = st.sidebar.number_input('Release Clause (EUR)', 1000, 500000000, 1000000)
    dribbling = st.sidebar.slider('Dribbling', 10, 100, 50)
    wage_eur = st.sidebar.number_input('Wage (EUR)', 500, 500000, 10000)
    shot_power = st.sidebar.slider('Power Shot Power', 10, 100, 50)
    value_eur = st.sidebar.number_input('Value (EUR)', 1000, 500000000, 1000000)
    vision = st.sidebar.slider('Mentality Vision', 10, 100, 50)
    short_passing = st.sidebar.slider('Attacking Short Passing', 10, 100, 50)
    physic = st.sidebar.slider('Physic', 10, 100, 50)
    long_passing = st.sidebar.slider('Skill Long Passing', 10, 100, 50)
    age = st.sidebar.slider('Age', 16, 40, 25)
    shooting = st.sidebar.slider('Shooting', 10, 100, 50)
    ball_control = st.sidebar.slider('Skill Ball Control', 10, 100, 50)
    real_face = st.sidebar.selectbox('Real Face', [0, 1], 0)
    international_reputation = st.sidebar.slider('International Reputation', 1, 5, 1)
    skill_curve = st.sidebar.slider('Skill Curve', 10, 100, 50)
    crossing = st.sidebar.slider('Attacking Crossing', 10, 100, 50)
    long_shots = st.sidebar.slider('Power Long Shots', 10, 100, 50)
    aggression = st.sidebar.slider('Mentality Aggression', 10, 100, 50)
    
    data = {
        'movement_reactions': reactions,
        'mentality_composure': composure,
        'passing': passing,
        'potential': potential,
        'release_clause_eur': release_clause_eur,
        'dribbling': dribbling,
        'wage_eur': wage_eur,
        'power_shot_power': shot_power,
        'value_eur': value_eur,
        'mentality_vision': vision,
        'attacking_short_passing': short_passing,
        'physic': physic,
        'skill_long_passing': long_passing,
        'age': age,
        'shooting': shooting,
        'skill_ball_control': ball_control,
        'real_face': real_face,
        'international_reputation': international_reputation,
        'skill_curve': skill_curve,
        'attacking_crossing': crossing,
        'power_long_shots': long_shots,
        'mentality_aggression': aggression
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input()

# Display user input
st.subheader('User Input')
st.write(input_df)

# Predict button
if st.button('Predict'):
    # Predict and display the output
    predicted_rating = model.predict(scaler.transform(input_df))
    st.subheader('Predicted Rating')
    st.write(np.floor(predicted_rating[0]))







