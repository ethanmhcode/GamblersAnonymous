import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.metric_cards import style_metric_cards

# Set Streamlit page configuration
st.set_page_config(page_title="NBA Player Performance Prediction", layout="wide")

# Customizing UI Theme
st.markdown(
    """
    <style>
        body { font-family: 'Arial', sans-serif; }
        .stButton>button { background-color: #FF5733; color: white; font-weight: bold; }
        .stButton>button:hover { background-color: #FF5733; color: black; }
        .stSelectbox>div>div { border-radius: 10px; }
        .stDataFrame { border-radius: 10px; }
        .stMarkdown { font-size: 16px; }
    </style>
    """,
    unsafe_allow_html=True
)

# Load dataset
df = pd.read_csv("nba_active_players_game_logs_2021_24.csv")

# Defensive ratings for all 30 teams
defensive_ratings = {
    "Oklahoma City Thunder": 107.0,
    "Orlando Magic": 109.6,
    "Houston Rockets": 110.1,
    "LA Clippers": 110.4,
    "Memphis Grizzlies": 110.7,
    "Boston Celtics": 111.2,
    "Minnesota Timberwolves": 111.8,
    "Milwaukee Bucks": 112.0,
    "Cleveland Cavaliers": 112.1,
    "Golden State Warriors": 112.4,
    "Miami Heat": 112.8,
    "Dallas Mavericks": 112.9,
    "Sacramento Kings": 113.0,
    "Detroit Pistons": 113.1,
    "Charlotte Hornets": 113.2,
    "New York Knicks": 114.4,
    "Denver Nuggets": 114.5,
    "Atlanta Hawks": 114.6,
    "Indiana Pacers": 114.7,
    "San Antonio Spurs": 114.8,
    "Philadelphia 76ers": 115.0,
    "LA Lakers": 115.1,
    "Chicago Bulls": 115.2,
    "Brooklyn Nets": 115.3,
    "Phoenix Suns": 115.4,
    "Portland Trail Blazers": 115.5,
    "Toronto Raptors": 115.6,
    "New Orleans Pelicans": 115.7,
    "Utah Jazz": 115.9,
    "Washington Wizards": 116.2
}

# Convert defensive ratings dictionary to DataFrame
defensive_ratings_df = pd.DataFrame(list(defensive_ratings.items()), columns=["TEAM", "DEF_RATING"])

# Merge defensive ratings into the main DataFrame based on team name
df = pd.merge(df, defensive_ratings_df, how="left", left_on="TEAM", right_on="TEAM")

# Ensure categorical columns are properly formatted
categorical_cols = ["MATCHUP", "TEAM", "PLAYER_NAME"]
label_encoders = {}

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[f"{col}_original"] = df[col]  # Store original values before encoding
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    else:
        st.warning(f"Warning: {col} column not found in dataset")

# Convert GAME_DATE to datetime and sort
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
df = df.sort_values(by=['PLAYER_NAME', 'GAME_DATE'])

# Compute rolling averages for recent games (last 5 games)
rolling_window = 5
df['PTS_L5G'] = df.groupby('PLAYER_NAME')['PTS'].transform(lambda x: x.rolling(rolling_window, min_periods=1).mean())
df['REB_L5G'] = df.groupby('PLAYER_NAME')['REB'].transform(lambda x: x.rolling(rolling_window, min_periods=1).mean())
df['AST_L5G'] = df.groupby('PLAYER_NAME')['AST'].transform(lambda x: x.rolling(rolling_window, min_periods=1).mean())

# Define features and targets
features = ["MATCHUP", "TEAM", "PLAYER_NAME", "FGM", "FGA", "FTM", "FTA", "OREB", "DREB", "TOV", "PTS_L5G", "REB_L5G", "AST_L5G", "DEF_RATING"]
targets = ["PTS", "REB", "AST", "3PM", "BLK", "STL"]

# Ensure features exist in dataset
features = [col for col in features if col in df.columns]

# Drop rows with missing target values
df = df.dropna(subset=targets)
X = df[features]
y = df[targets]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6)
xgb_model.fit(X_train, y_train)

# Streamlit UI
st.title("🏀 NBA Player Performance Prediction")
st.subheader("Predict player performance for upcoming matchups.")
st.divider()

# Add description about the app
st.markdown("""
    ### About the App
    This app predicts the performance of NBA players for upcoming matchups based on historical data and various statistical metrics. 
    It leverages machine learning models, specifically XGBoost, to forecast a player's points (PTS), rebounds (REB), assists (AST), three-pointers made (3PM), blocks (BLK), and steals (STL) in a given game.

    #### How It Works:
    1. **Data Input**: The model uses historical game logs, including statistics like field goals made, assists, rebounds, and defensive ratings of the teams.
    2. **Recent Performance**: It calculates rolling averages for each player's recent games to capture trends and performance patterns.
    3. **Model Training**: An XGBoost regression model is trained on the data to predict the player's performance in upcoming games based on their recent stats and opponent's defensive ratings.
    4. **Results**: The app provides predictions for a player's performance and compares it to their averages over the last 5 games, helping you make informed decisions.

    Try selecting a player and an opponent team to see how the app forecasts the player's upcoming performance!
""")

# Side-by-side player and opponent selection
col1, col2 = st.columns(2)
players = sorted(df["PLAYER_NAME_original"].unique())
teams = sorted(df["TEAM_original"].unique())

with col1:
    selected_player = st.selectbox("Select Player", players)
with col2:
    selected_team = st.selectbox("Select Opponent Team", teams)

# Prediction logic
def predict_player_performance(player_name, opponent_team):
    def encode_value(encoder, value):
        return encoder.transform([value])[0] if value in encoder.classes_ else -1

    player_encoded = encode_value(label_encoders["PLAYER_NAME"], player_name)
    team_encoded = encode_value(label_encoders["TEAM"], opponent_team)

    matchup_games = df[(df["PLAYER_NAME_original"] == player_name) & df["MATCHUP_original"].str.contains(opponent_team)]

    if matchup_games.empty:
        st.warning(f"No recent games found for {player_name} against {opponent_team}. Using default values.")
        return pd.DataFrame({"Stat": ["PTS", "REB", "AST", "3PM", "BLK", "STL"], "Prediction": [0, 0, 0, 0, 0, 0], "Average": [0, 0, 0, 0, 0, 0]})

    recent_games = matchup_games.iloc[-1]

    sample_input_values = [recent_games[col] if col in recent_games else 0 for col in X_train.columns]
    sample_input = pd.DataFrame([sample_input_values], columns=X_train.columns)
    predicted_stats = xgb_model.predict(sample_input)[0]

    predicted_df = pd.DataFrame({
        "Stat": ["PTS", "REB", "AST", "3PM", "BLK", "STL"],
        "Prediction": [round(predicted_stats[0], 1), round(predicted_stats[1], 1), round(predicted_stats[2], 1),
                       round(predicted_stats[3], 1), round(predicted_stats[4], 1), round(predicted_stats[5], 1)]
    })

    return predicted_df

def get_player_averages(player_name):
    # Get the player's last 5 games for averaging
    player_games = df[df["PLAYER_NAME_original"] == player_name].tail(5)
    avg_pts = player_games["PTS"].mean()
    avg_reb = player_games["REB"].mean()
    avg_ast = player_games["AST"].mean()
    avg_3pm = player_games["3PM"].mean()
    avg_blk = player_games["BLK"].mean()
    avg_stl = player_games["STL"].mean()

    avg_df = pd.DataFrame({
        "Stat": ["PTS", "REB", "AST", "3PM", "BLK", "STL"],
        "Average": [round(avg_pts, 1), round(avg_reb, 1), round(avg_ast, 1),
                    round(avg_3pm, 1), round(avg_blk, 1), round(avg_stl, 1)]
    })

    return avg_df

def plot_comparative_line_chart(average_df, predicted_df):
    # Merge the two dataframes on 'Stat' for comparison
    comparison_df = pd.merge(average_df, predicted_df, on='Stat')

    # Create the interactive Plotly line chart
    fig = go.Figure()

    # Add trace for the player's averages
    fig.add_trace(go.Scatter(
        x=comparison_df['Stat'], y=comparison_df['Average'],
        mode='lines+markers', name='Average', line=dict(color='blue', width=2)
    ))

    # Add trace for the predicted stats
    fig.add_trace(go.Scatter(
        x=comparison_df['Stat'], y=comparison_df['Prediction'],
        mode='lines+markers', name='Predicted', line=dict(color='orange', width=2, dash='dash')
    ))

    # Update layout for better visualization
    fig.update_layout(
        title=f'Comparison of {selected_player} Averages vs Predicted Results',
        xaxis_title='Stat',
        yaxis_title='Stat Value',
        legend_title='Legend',
        hovermode='closest'
    )

    # Show the interactive chart in Streamlit
    st.plotly_chart(fig)

# Display prediction button and output
if st.button("🔮 Predict Performance", use_container_width=True):
    prediction_df = predict_player_performance(selected_player, selected_team)
    average_df = get_player_averages(selected_player)

    # Merge player averages with the prediction
    combined_df = pd.merge(prediction_df, average_df, on='Stat')

    st.write(f"### Predicted Stats and Player Averages for {selected_player} vs {selected_team}")
    st.dataframe(combined_df)

    # Display comparative line chart
    plot_comparative_line_chart(average_df, prediction_df)
