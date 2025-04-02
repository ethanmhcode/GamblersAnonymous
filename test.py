import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from streamlit_extras.metric_cards import style_metric_cards

# Set Streamlit page configuration
st.set_page_config(page_title="NBA Player Performance Prediction", layout="wide")

# Cache dataset loading (use Parquet for faster loading)
@st.cache_data
def load_data():
    # If CSV, convert to Parquet first (done once)
    df = pd.read_csv("nba_data.csv")  # Keeping CSV file for data path
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(by=['PLAYER_NAME', 'GAME_DATE'])
    
    # Convert categorical columns to 'category' dtype to save memory
    for col in ['PLAYER_NAME', 'TEAM', 'MATCHUP']:
        df[col] = df[col].astype('category')
        
    return df

df = load_data()

# Defensive ratings dictionary
DEF_RATINGS = {
    "Oklahoma City Thunder": 107.0, "Orlando Magic": 109.6, "Houston Rockets": 110.1, "LA Clippers": 110.4,
    "Memphis Grizzlies": 110.7, "Boston Celtics": 111.2, "Minnesota Timberwolves": 111.8, "Milwaukee Bucks": 112.0,
    "Cleveland Cavaliers": 112.1, "Golden State Warriors": 112.4, "Miami Heat": 112.8, "Dallas Mavericks": 112.9,
    "Sacramento Kings": 113.0, "Detroit Pistons": 113.1, "Charlotte Hornets": 113.2, "New York Knicks": 114.4,
    "Denver Nuggets": 114.5, "Atlanta Hawks": 114.6, "Indiana Pacers": 114.7, "San Antonio Spurs": 114.8,
    "Philadelphia 76ers": 115.0, "LA Lakers": 115.1, "Chicago Bulls": 115.2, "Brooklyn Nets": 115.3,
    "Phoenix Suns": 115.4, "Portland Trail Blazers": 115.5, "Toronto Raptors": 115.6, "New Orleans Pelicans": 115.7,
    "Utah Jazz": 115.9, "Washington Wizards": 116.2
}

df['DEF_RATING'] = df['TEAM'].map(DEF_RATINGS)

# Encode categorical columns
@st.cache_data
def encode_columns(df):
    label_encoders = {}
    categorical_cols = ["MATCHUP", "TEAM", "PLAYER_NAME"]
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[f"{col}_encoded"] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    return df, label_encoders

df, label_encoders = encode_columns(df)

# Optimized rolling averages using vectorized operations
ROLLING_WINDOW = 5
for col in ['PTS', 'REB', 'AST']:
    df[f'{col}_L5G'] = df.groupby('PLAYER_NAME')[col].apply(lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean())

# Define features and targets
FEATURES = ["MATCHUP_encoded", "TEAM_encoded", "PLAYER_NAME_encoded", "FGM", "FGA", "FTM", "FTA", "OREB", "DREB", "TOV", "PTS_L5G", "REB_L5G", "AST_L5G", "DEF_RATING"]
TARGETS = ["PTS", "REB", "AST", "3PM", "BLK", "STL"]

# Prepare data
X = df[FEATURES].dropna()
y = df.loc[X.index, TARGETS]

# Train model (cached)
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6)
    model.fit(X_train, y_train)
    return model

xgb_model = train_model(X, y)

# UI
st.title("🏀 NBA Player Performance Prediction")
st.subheader("Predict player performance for upcoming matchups.")
st.divider()

players = sorted(df["PLAYER_NAME"].unique())
teams = sorted(df["TEAM"].unique())

col1, col2 = st.columns(2)
with col1:
    selected_player = st.selectbox("Select Player", players)
with col2:
    selected_team = st.selectbox("Select Opponent Team", teams)

def predict_player_performance(player_name, opponent_team):
    player_encoded = label_encoders["PLAYER_NAME"].transform([player_name])[0]
    team_encoded = label_encoders["TEAM"].transform([opponent_team])[0]
    
    matchup_games = df[(df["PLAYER_NAME"] == player_name) & df["MATCHUP"].str.contains(opponent_team)]
    if matchup_games.empty:
        return pd.DataFrame({"Stat": TARGETS, "Prediction": [0] * len(TARGETS)})
    
    recent_game = matchup_games.iloc[-1]
    sample_input = pd.DataFrame([[recent_game[col] if col in recent_game else 0 for col in FEATURES]], columns=FEATURES)
    predictions = xgb_model.predict(sample_input)[0]
    return pd.DataFrame({"Stat": TARGETS, "Prediction": predictions})

def get_player_averages(player_name):
    player_games = df[df["PLAYER_NAME"] == player_name].tail(5)
    averages = player_games[TARGETS].mean()
    return pd.DataFrame({"Stat": TARGETS, "Average": averages})

def plot_comparative_chart(average_df, predicted_df):
    comparison_df = pd.merge(average_df, predicted_df, on='Stat')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=comparison_df['Stat'], y=comparison_df['Average'], mode='lines+markers', name='Average', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=comparison_df['Stat'], y=comparison_df['Prediction'], mode='lines+markers', name='Predicted', line=dict(color='orange', width=2, dash='dash')))
    fig.update_layout(title=f'Comparison of {selected_player} Averages vs Predictions', xaxis_title='Stat', yaxis_title='Value', hovermode='closest')
    st.plotly_chart(fig)

if st.button("🔮 Predict Performance", use_container_width=True):
    prediction_df = predict_player_performance(selected_player, selected_team)
    average_df = get_player_averages(selected_player)
    combined_df = pd.merge(prediction_df, average_df, on='Stat')
    st.write(f"### Predicted Stats vs Averages for {selected_player} vs {selected_team}")
    st.dataframe(combined_df)
    plot_comparative_chart(average_df, prediction_df)
