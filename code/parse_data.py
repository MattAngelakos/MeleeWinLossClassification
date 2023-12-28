import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# Specify the path to your CSV file
csv_file_path = 'C:/Users/squir/Downloads/archive/melee_player_database/sets.csv'
csv_file_path_clean = 'C:/Users/squir/Downloads/archive/melee_player_database/sets_cleaned.csv'
csv_file_path_clean_num = 'C:/Users/squir/Downloads/archive/melee_player_database/sets_cleaned_num.csv'
# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)
df=df[df['game_data'] != '[]']
df = df.drop(columns=['winner_id'])
df['game_data'] = df['game_data'].apply(json.loads)
# Explode the 'game_data' column and reset the index
df_exploded = df.explode('game_data').reset_index(drop=True)

# Use json_normalize to flatten the 'game_data' column
df_normalized = pd.json_normalize(df_exploded['game_data'])

# Concatenate the normalized DataFrame with the original DataFrame
df = pd.concat([df_exploded, df_normalized], axis=1)
# Drop the original 'game_data' column
df = df.drop(columns=['game_data', 'index', 'key', 'game', 'p1_score', 'p2_score', 'winner_score', 'loser_score'])
# Drop the original columns
df['location_names'] = df['location_names'].apply(lambda x: x.split(', ')[1] if isinstance(x, str) else x)
df['bracket_name'] = df['bracket_name'].fillna('Bracket')
df['winner_char'] = df['winner_char'].str.replace('melee/', '')
df['loser_char'] = df['loser_char'].str.replace('melee/', '')
#print(df)
#set default values for the new columns
df['win_status'] = 'win'
df['p1_char'] = ' '
df['p2_char'] = ' '
#remove na values in order to transform data to the proper type
df.dropna(subset=['p1_id'], inplace=True)
df.dropna(subset=['loser_id'], inplace=True)
df.dropna(subset=['winner_id'], inplace=True)
#change columns to the proper data type
df['p1_id'] = df['p1_id'].astype(int)
df['p2_id'] = df['p2_id'].astype(int)
df['bracket_order'] = df['bracket_order'].astype(int)
df['best_of'] = df['best_of'].astype(int)
df['loser_id'] = df['loser_id'].astype(int)
df['winner_id'] = df['winner_id'].astype(int)
#we create a mask to find all the locations where the loser is player 1
#at these locations we set the win_status to lose as well as making the player 1 and 2 character's the correct one
mask = (df['loser_id'] == df['p1_id'])
df.loc[mask, 'win_status'] = 'lose'
df.loc[mask, 'p1_char'] = df.loc[mask, 'loser_char']
df.loc[mask, 'p2_char'] = df.loc[mask, 'winner_char']
#do the same for when player 1 wins
mask2 = (df['winner_id'] == df['p1_id'])
df.loc[mask2, 'win_status'] = 'win'
df.loc[mask2, 'p1_char'] = df.loc[mask2, 'winner_char']
df.loc[mask2, 'p2_char'] = df.loc[mask2, 'loser_char']
#remove the old columns 
df = df.drop(columns=['winner_char', 'loser_char', 'winner_id', 'loser_id'])
#update the naming of the columns for readability
df.columns = ['tournament_key', 'player_id', 'opponent_id', 'location_names', 'bracket_name', 'bracket_order', 'set_order', 'best_of', 'stage', 'win_status', 'player_char', 'opponent_char']
df.dropna(subset=['stage'], inplace=True)
df.dropna(subset=['player_char'], inplace=True)
df.dropna(subset=['opponent_char'], inplace=True)
df.dropna(subset=['tournament_key'], inplace=True)
df.dropna(subset=['location_names'], inplace=True)
df.dropna(subset=['bracket_name'], inplace=True)
df.dropna(subset=['bracket_order'], inplace=True)
df.dropna(subset=['set_order'], inplace=True)
df.dropna(subset=['best_of'], inplace=True)
df.dropna(subset=['player_id'], inplace=True)
df.dropna(subset=['opponent_id'], inplace=True)
#print(df)
df.to_csv(csv_file_path_clean, index=False)
le = LabelEncoder()
df['tournament_key'] = le.fit_transform(df['tournament_key'])
df['location_names'] = le.fit_transform(df['location_names'])
df['bracket_name'] = le.fit_transform(df['bracket_name'])
df['bracket_order'] = le.fit_transform(df['bracket_order'])
df['set_order'] = le.fit_transform(df['set_order'])
df['best_of'] = le.fit_transform(df['best_of'])
df['player_char'] = le.fit_transform(df['player_char'])
df['opponent_char'] = le.fit_transform(df['opponent_char'])
df['player_id'] = le.fit_transform(df['player_id'])
df['opponent_id'] = le.fit_transform(df['opponent_id'])
df['stage'] = le.fit_transform(df['stage'])
print(df)
df.to_csv(csv_file_path_clean_num, index=False)
# Display the modified dataframe
df = pd.read_csv(csv_file_path_clean)
print(df[df.isna().any(axis=1)]) # shows NaN values in data frame if it exists
print(df)
