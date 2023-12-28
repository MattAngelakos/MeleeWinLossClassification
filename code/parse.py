import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
# Specify the path to your CSV file
csv_file_path = 'C:/Users/squir/Downloads/archive/melee_player_database/sets.csv'
csv_file_path_clean = 'C:/Users/squir/Downloads/archive/melee_player_database/sets_cleaned.csv'
csv_file_path_clean_num = 'C:/Users/squir/Downloads/archive/melee_player_database/sets_cleaned_num.csv'
# Display the modified dataframe
df = pd.read_csv(csv_file_path_clean)
print(df[df.isna().any(axis=1)]) # shows NaN values in data frame if it exists
df = df.dropna()
print((df['win_status'] == 'lose').sum())
print((df['win_status'] == 'win').sum())
outcome = {
    'Win': (df['win_status'] == 'win').sum(),
    'Loss': (df['win_status'] == 'lose').sum(),
}

# Convert the dictionary to lists
outcomes = list(outcome.keys())
amount = list(outcome.values())
# Create a bar graph
plt.bar(outcomes, amount)
plt.title('Balance of Win Loss Data')
plt.xlabel('Outcome')
plt.ylabel('Number of Games')
# Show the plot
plt.show()
print(df)
df = pd.read_csv(csv_file_path_clean_num)
attr = df.drop('win_status', axis=1)
target = df['win_status']
games_played_vs_winrate =[]
wins = ((df['player_char'] == 'fox') & (df['win_status'] == 'win')).sum()
losses = ((df['player_char'] == 'fox') & (df['win_status'] == 'lose')).sum()
#dittos = ((df['player_char'] == 'fox') & (df['opponent_char'] == 'fox')).sum()
winrate = (wins)/(wins+losses)
matches = wins+losses
print("")
print("fox")
print(matches)
print(winrate)
games_played_vs_winrate.append([matches, winrate])
wins = ((df['player_char'] == 'marth') & (df['win_status'] == 'win')).sum()
losses = ((df['player_char'] == 'marth') & (df['win_status'] == 'lose')).sum()
#dittos = ((df['player_char'] == 'marth') & (df['opponent_char'] == 'marth')).sum()
winrate = (wins)/(wins+losses)
matches = wins+losses
print("")
print("marth")
print(matches)
print(winrate)
games_played_vs_winrate.append([matches, winrate])
wins = ((df['player_char'] == 'jigglypuff') & (df['win_status'] == 'win')).sum()
losses = ((df['player_char'] == 'jigglypuff') & (df['win_status'] == 'lose')).sum()
#dittos = ((df['player_char'] == 'jigglypuff') & (df['opponent_char'] == 'jigglypuff')).sum()
winrate = (wins)/(wins+losses)
matches = wins+losses
print("")
print("jigglypuff")
print(matches)
print(winrate)
games_played_vs_winrate.append([matches, winrate])
wins = ((df['player_char'] == 'falco') & (df['win_status'] == 'win')).sum()
losses = ((df['player_char'] == 'falco') & (df['win_status'] == 'lose')).sum()
#dittos = ((df['player_char'] == 'falco') & (df['opponent_char'] == 'falco')).sum()
winrate = (wins)/(wins+losses)
matches = wins+losses
print("")
print("falco")
print(matches)
print(winrate)
games_played_vs_winrate.append([matches, winrate])
wins = ((df['player_char'] == 'sheik') & (df['win_status'] == 'win')).sum()
losses = ((df['player_char'] == 'sheik') & (df['win_status'] == 'lose')).sum()
#dittos = ((df['player_char'] == 'sheik') & (df['opponent_char'] == 'sheik')).sum()
winrate = (wins)/(wins+losses)
matches = wins+losses
print("")
print("sheik")
print(matches)
print(winrate)
games_played_vs_winrate.append([matches, winrate])
wins = ((df['player_char'] == 'captainfalcon') & (df['win_status'] == 'win')).sum()
losses = ((df['player_char'] == 'captainfalcon') & (df['win_status'] == 'lose')).sum()
#dittos = ((df['player_char'] == 'captainfalcon') & (df['opponent_char'] == 'captainfalcon')).sum()
winrate = (wins)/(wins+losses)
matches = wins+losses
print("")
print("captainfalcon")
print(matches)
print(winrate)
games_played_vs_winrate.append([matches, winrate])
wins = ((df['player_char'] == 'peach') & (df['win_status'] == 'win')).sum()
losses = ((df['player_char'] == 'peach') & (df['win_status'] == 'lose')).sum()
#dittos = ((df['player_char'] == 'peach') & (df['opponent_char'] == 'peach')).sum()
winrate = (wins)/(wins+losses)
matches = wins+losses
print("")
print("peach")
print(matches)
print(winrate)
games_played_vs_winrate.append([matches, winrate])
wins = ((df['player_char'] == 'iceclimbers') & (df['win_status'] == 'win')).sum()
losses = ((df['player_char'] == 'iceclimbers') & (df['win_status'] == 'lose')).sum()
#dittos = ((df['player_char'] == 'iceclimbers') & (df['opponent_char'] == 'iceclimbers')).sum()
winrate = (wins)/(wins+losses)
matches = wins+losses
print("")
print("iceclimbers")
print(matches)
print(winrate)
games_played_vs_winrate.append([matches, winrate])
wins = ((df['player_char'] == 'pikachu') & (df['win_status'] == 'win')).sum()
losses = ((df['player_char'] == 'pikachu') & (df['win_status'] == 'lose')).sum()
#dittos = ((df['player_char'] == 'pikachu') & (df['opponent_char'] == 'pikachu')).sum()
winrate = (wins)/(wins+losses)
matches = wins+losses
print("")
print("pikachu")
print(matches)
print(winrate)
games_played_vs_winrate.append([matches, winrate])
wins = ((df['player_char'] == 'yoshi') & (df['win_status'] == 'win')).sum()
losses = ((df['player_char'] == 'yoshi') & (df['win_status'] == 'lose')).sum()
#dittos = ((df['player_char'] == 'yoshi') & (df['opponent_char'] == 'yoshi')).sum()
winrate = (wins)/(wins+losses)
matches = wins+losses
print("")
print("yoshi")
print(matches)
print(winrate)
games_played_vs_winrate.append([matches, winrate])
wins = ((df['player_char'] == 'samus') & (df['win_status'] == 'win')).sum()
losses = ((df['player_char'] == 'samus') & (df['win_status'] == 'lose')).sum()
#dittos = ((df['player_char'] == 'samus') & (df['opponent_char'] == 'samus')).sum()
winrate = (wins)/(wins+losses)
matches = wins+losses
print("")
print("samus")
print(matches)
print(winrate)
games_played_vs_winrate.append([matches, winrate])
wins = ((df['player_char'] == 'luigi') & (df['win_status'] == 'win')).sum()
losses = ((df['player_char'] == 'luigi') & (df['win_status'] == 'lose')).sum()
#dittos = ((df['player_char'] == 'luigi') & (df['opponent_char'] == 'luigi')).sum()
winrate = (wins)/(wins+losses)
matches = wins+losses
print("")
print("luigi")
print(matches)
print(winrate)
games_played_vs_winrate.append([matches, winrate])
wins = ((df['player_char'] == 'drmario') & (df['win_status'] == 'win')).sum()
losses = ((df['player_char'] == 'drmario') & (df['win_status'] == 'lose')).sum()
#dittos = ((df['player_char'] == 'drmario') & (df['opponent_char'] == 'drmario')).sum()
winrate = (wins)/(wins+losses)
matches = wins+losses
print("")
print("drmario")
print(matches)
print(winrate)
games_played_vs_winrate.append([matches, winrate])
wins = ((df['player_char'] == 'ganondorf') & (df['win_status'] == 'win')).sum()
losses = ((df['player_char'] == 'ganondorf') & (df['win_status'] == 'lose')).sum()
#dittos = ((df['player_char'] == 'ganondorf') & (df['opponent_char'] == 'ganondorf')).sum()
winrate = (wins)/(wins+losses)
matches = wins+losses
print("")
print("ganondorf")
print(matches)
print(winrate)
games_played_vs_winrate.append([matches, winrate])
wins = ((df['player_char'] == 'mario') & (df['win_status'] == 'win')).sum()
losses = ((df['player_char'] == 'mario') & (df['win_status'] == 'lose')).sum()
#dittos = ((df['player_char'] == 'mario') & (df['opponent_char'] == 'mario')).sum()
winrate = (wins)/(wins+losses)
matches = wins+losses
print("")
print("mario")
print(matches)
print(winrate)
games_played_vs_winrate.append([matches, winrate])
wins = ((df['player_char'] == 'donkeykong') & (df['win_status'] == 'win')).sum()
losses = ((df['player_char'] == 'donkeykong') & (df['win_status'] == 'lose')).sum()
#dittos = ((df['player_char'] == 'donkeykong') & (df['opponent_char'] == 'donkeykong')).sum()
winrate = (wins)/(wins+losses)
matches = wins+losses
print("")
print("donkeykong")
print(matches)
print(winrate)
games_played_vs_winrate.append([matches, winrate])
wins = ((df['player_char'] == 'younglink') & (df['win_status'] == 'win')).sum()
losses = ((df['player_char'] == 'younglink') & (df['win_status'] == 'lose')).sum()
#dittos = ((df['player_char'] == 'younglink') & (df['opponent_char'] == 'younglink')).sum()
winrate = (wins)/(wins+losses)
matches = wins+losses
print("")
print("younglink")
print(matches)
print(winrate)
games_played_vs_winrate.append([matches, winrate])
wins = ((df['player_char'] == 'link') & (df['win_status'] == 'win')).sum()
losses = ((df['player_char'] == 'link') & (df['win_status'] == 'lose')).sum()
#dittos = ((df['player_char'] == 'link') & (df['opponent_char'] == 'link')).sum()
winrate = (wins)/(wins+losses)
matches = wins+losses
print("")
print("link")
print(matches)
print(winrate)
games_played_vs_winrate.append([matches, winrate])
wins = ((df['player_char'] == 'mrgameandwatch') & (df['win_status'] == 'win')).sum()
losses = ((df['player_char'] == 'mrgameandwatch') & (df['win_status'] == 'lose')).sum()
#dittos = ((df['player_char'] == 'mrgameandwatch') & (df['opponent_char'] == 'mrgameandwatch')).sum()
winrate = (wins)/(wins+losses)
matches = wins+losses
print("")
print("mrgameandwatch")
print(matches)
print(winrate)
games_played_vs_winrate.append([matches, winrate])
wins = ((df['player_char'] == 'mewtwo') & (df['win_status'] == 'win')).sum()
losses = ((df['player_char'] == 'mewtwo') & (df['win_status'] == 'lose')).sum()
#dittos = ((df['player_char'] == 'mewtwo') & (df['opponent_char'] == 'mewtwo')).sum()
winrate = (wins)/(wins+losses)
matches = wins+losses
print("")
print("mewtwo")
print(matches)
print(winrate)
games_played_vs_winrate.append([matches, winrate])
wins = ((df['player_char'] == 'roy') & (df['win_status'] == 'win')).sum()
losses = ((df['player_char'] == 'roy') & (df['win_status'] == 'lose')).sum()
#dittos = ((df['player_char'] == 'roy') & (df['opponent_char'] == 'roy')).sum()
winrate = (wins)/(wins+losses)
matches = wins+losses
print("")
print("roy")
print(matches)
print(winrate)
games_played_vs_winrate.append([matches, winrate])
wins = ((df['player_char'] == 'pichu') & (df['win_status'] == 'win')).sum()
losses = ((df['player_char'] == 'pichu') & (df['win_status'] == 'lose')).sum()
#dittos = ((df['player_char'] == 'pichu') & (df['opponent_char'] == 'pichu')).sum()
winrate = (wins)/(wins+losses)
matches = wins+losses
print("")
print("pichu")
print(matches)
print(winrate)
games_played_vs_winrate.append([matches, winrate])
wins = ((df['player_char'] == 'ness') & (df['win_status'] == 'win')).sum()
losses = ((df['player_char'] == 'ness') & (df['win_status'] == 'lose')).sum()
#dittos = ((df['player_char'] == 'ness') & (df['opponent_char'] == 'ness')).sum()
winrate = (wins)/(wins+losses)
matches = wins+losses
print("")
print("ness")
print(matches)
print(winrate)
games_played_vs_winrate.append([matches, winrate])
wins = ((df['player_char'] == 'zelda') & (df['win_status'] == 'win')).sum()
losses = ((df['player_char'] == 'zelda') & (df['win_status'] == 'lose')).sum()
#dittos = ((df['player_char'] == 'zelda') & (df['opponent_char'] == 'zelda')).sum()
winrate = (wins)/(wins+losses)
matches = wins+losses
print("")
print("zelda")
print(matches)
print(winrate)
games_played_vs_winrate.append([matches, winrate])
wins = ((df['player_char'] == 'kirby') & (df['win_status'] == 'win')).sum()
losses = ((df['player_char'] == 'kirby') & (df['win_status'] == 'lose')).sum()
#dittos = ((df['player_char'] == 'kirby') & (df['opponent_char'] == 'kirby')).sum()
winrate = (wins)/(wins+losses)
matches = wins+losses
print("")
print("kirby")
print(matches)
print(winrate)
games_played_vs_winrate.append([matches, winrate])
wins = ((df['player_char'] == 'bowser') & (df['win_status'] == 'win')).sum()
losses = ((df['player_char'] == 'bowser') & (df['win_status'] == 'lose')).sum()
#dittos = ((df['player_char'] == 'bowser') & (df['opponent_char'] == 'bowser')).sum()
winrate = (wins)/(wins+losses)
matches = wins+losses
print("")
print("bowser")
print(matches)
print(winrate)
games_played_vs_winrate.append([matches, winrate])
x_coordinates = [point[0] for point in games_played_vs_winrate]
y_coordinates = [point[1] for point in games_played_vs_winrate]
plt.scatter(x_coordinates, y_coordinates, marker='o', color='blue')
plt.text(x_coordinates[0], y_coordinates[0], 'fox', fontsize=8, ha='right', va='bottom')
plt.text(x_coordinates[1], y_coordinates[1], 'marth', fontsize=8, ha='right', va='bottom')
plt.text(x_coordinates[2], y_coordinates[2], 'jigglypuff', fontsize=8, ha='right', va='bottom')
plt.text(x_coordinates[3], y_coordinates[3], 'falco', fontsize=8, ha='right', va='bottom')
plt.text(x_coordinates[4], y_coordinates[4], 'sheik', fontsize=8, ha='right', va='bottom')
plt.text(x_coordinates[5], y_coordinates[5], 'captainfalcon', fontsize=8, ha='right', va='bottom')
plt.text(x_coordinates[6], y_coordinates[6], 'peach', fontsize=8, ha='right', va='bottom')
plt.text(x_coordinates[7], y_coordinates[7], 'iceclimbers', fontsize=8, ha='right', va='bottom')
plt.text(x_coordinates[8], y_coordinates[8], 'pikachu', fontsize=8, ha='right', va='bottom')
plt.text(x_coordinates[9], y_coordinates[9], 'yoshi', fontsize=8, ha='right', va='bottom')
plt.text(x_coordinates[10], y_coordinates[10], 'samus', fontsize=8, ha='right', va='bottom')
plt.text(x_coordinates[11], y_coordinates[11], 'luigi', fontsize=8, ha='right', va='bottom')
plt.text(x_coordinates[12], y_coordinates[12], 'drmario', fontsize=8, ha='right', va='bottom')
plt.text(x_coordinates[13], y_coordinates[13], 'ganondorf', fontsize=8, ha='right', va='bottom')
plt.text(x_coordinates[14], y_coordinates[14], 'mario', fontsize=8, ha='right', va='bottom')
plt.text(x_coordinates[15], y_coordinates[15], 'donkeykong', fontsize=8, ha='right', va='bottom')
plt.text(x_coordinates[16], y_coordinates[16], 'younglink', fontsize=8, ha='right', va='bottom')
plt.text(x_coordinates[17], y_coordinates[17], 'link', fontsize=8, ha='right', va='bottom')
plt.text(x_coordinates[18], y_coordinates[18], 'mrgameandwatch', fontsize=8, ha='right', va='bottom')
plt.text(x_coordinates[19], y_coordinates[19], 'mewtwo', fontsize=8, ha='right', va='bottom')
plt.text(x_coordinates[20], y_coordinates[20], 'roy', fontsize=8, ha='right', va='bottom')
plt.text(x_coordinates[21], y_coordinates[21], 'pichu', fontsize=8, ha='right', va='bottom')
plt.text(x_coordinates[22], y_coordinates[22], 'ness', fontsize=8, ha='right', va='bottom')
plt.text(x_coordinates[23], y_coordinates[23], 'zelda', fontsize=8, ha='right', va='bottom')
plt.text(x_coordinates[24], y_coordinates[24], 'kirby', fontsize=8, ha='right', va='bottom')
plt.text(x_coordinates[25], y_coordinates[25], 'bowser', fontsize=8, ha='right', va='bottom')
plt.xlabel('Games Played')
plt.ylabel('Winrrate')
plt.title('Melee Characters Games Played vs Winrate')
plt.show()
plt.scatter(x_coordinates, y_coordinates, marker='o', color='blue')
plt.xscale('log')
plt.yscale('log')
plt.text(x_coordinates[0], y_coordinates[0], 'fox', fontsize=10, ha='right', va='bottom')
plt.text(x_coordinates[1], y_coordinates[1], 'marth', fontsize=10, ha='right', va='bottom')
plt.text(x_coordinates[2], y_coordinates[2], 'jigglypuff', fontsize=10, ha='right', va='bottom')
plt.text(x_coordinates[3], y_coordinates[3], 'falco', fontsize=10, ha='right', va='bottom')
plt.text(x_coordinates[4], y_coordinates[4], 'sheik', fontsize=10, ha='right', va='bottom')
plt.text(x_coordinates[5], y_coordinates[5], 'captainfalcon', fontsize=10, ha='right', va='bottom')
plt.text(x_coordinates[6], y_coordinates[6], 'peach', fontsize=10, ha='right', va='bottom')
plt.text(x_coordinates[7], y_coordinates[7], 'iceclimbers', fontsize=10, ha='right', va='bottom')
plt.text(x_coordinates[8], y_coordinates[8], 'pikachu', fontsize=10, ha='right', va='bottom')
plt.text(x_coordinates[9], y_coordinates[9], 'yoshi', fontsize=10, ha='right', va='bottom')
plt.text(x_coordinates[10], y_coordinates[10], 'samus', fontsize=10, ha='right', va='bottom')
plt.text(x_coordinates[11], y_coordinates[11], 'luigi', fontsize=10, ha='right', va='bottom')
plt.text(x_coordinates[12], y_coordinates[12], 'drmario', fontsize=10, ha='right', va='bottom')
plt.text(x_coordinates[13], y_coordinates[13], 'ganondorf', fontsize=10, ha='right', va='bottom')
plt.text(x_coordinates[14], y_coordinates[14], 'mario', fontsize=10, ha='right', va='bottom')
plt.text(x_coordinates[15], y_coordinates[15], 'donkeykong', fontsize=10, ha='right', va='bottom')
plt.text(x_coordinates[16], y_coordinates[16], 'younglink', fontsize=10, ha='right', va='bottom')
plt.text(x_coordinates[17], y_coordinates[17], 'link', fontsize=10, ha='right', va='bottom')
plt.text(x_coordinates[18], y_coordinates[18], 'mrgameandwatch', fontsize=10, ha='right', va='bottom')
plt.text(x_coordinates[19], y_coordinates[19], 'mewtwo', fontsize=10, ha='right', va='bottom')
plt.text(x_coordinates[20], y_coordinates[20], 'roy', fontsize=10, ha='right', va='bottom')
plt.text(x_coordinates[21], y_coordinates[21], 'pichu', fontsize=10, ha='right', va='bottom')
plt.text(x_coordinates[22], y_coordinates[22], 'ness', fontsize=10, ha='right', va='bottom')
plt.text(x_coordinates[23], y_coordinates[23], 'zelda', fontsize=10, ha='right', va='bottom')
plt.text(x_coordinates[24], y_coordinates[24], 'kirby', fontsize=10, ha='right', va='bottom')
plt.text(x_coordinates[25], y_coordinates[25], 'bowser', fontsize=10, ha='right', va='bottom')
plt.xlabel('Games Played (Log)')
plt.ylabel('Winrrate (Log)')
plt.title('Melee Characters Games Played vs Winrate (Log)')
plt.show()
games = {
    'Battlefield': (df['stage'] == 'Battlefield').sum() / 2,
    'Final Destination': (df['stage'] == 'Final Destination').sum() / 2,
    'Pokémon Stadium': (df['stage'] == 'Pokémon Stadium').sum() / 2,
    'Fountain of Dreams': (df['stage'] == 'Fountain of Dreams').sum() / 2,
    'Dream Land': (df['stage'] == 'Dream Land').sum() / 2,
    'Yoshi\'s Story': (df['stage'] == 'Yoshi\'s Story').sum() / 2,
}

# Convert the dictionary to lists
stages = list(games.keys())
num_games = list(games.values())

# Create a bar graph
plt.bar(stages, num_games)
plt.title('Number of Games Played on Each Stage')
plt.xlabel('Stage')
plt.ylabel('Number of Games')
# Show the plot
plt.show()
#logit
scaler = StandardScaler()
attr = pd.DataFrame(scaler.fit_transform(attr), columns=attr.columns)
attr_train, attr_test, target_train, target_test = train_test_split(attr, target, test_size=0.3, random_state=999, shuffle=True)
logit = LogisticRegression()
logit.fit(attr_train,target_train)
pred=logit.predict(attr_test)
target_pred=logit.predict(attr_test)
#C50
#attr = attr.drop(columns=['tournament_key', 'player_id', 'opponent_id'])
attr_train, attr_test, target_train, target_test = train_test_split(attr, target, test_size=0.2, random_state=7)
model = DecisionTreeClassifier(criterion='entropy', max_depth=300,splitter='best',max_leaf_nodes=500)
model.fit(attr_train,target_train)
target_pred = model.predict(attr_test)
#Forest
attr_train, attr_test, target_train, target_test = train_test_split(attr, target, test_size=0.2, random_state=7, shuffle=True)
model = RandomForestClassifier(n_estimators=100,random_state=7)
model.fit(attr_train,target_train)
target_pred = model.predict(attr_test)
#Tensorflow
attr_train, attr_test, target_train, target_test = train_test_split(attr, target, test_size=0.2, random_state=7)
scaler = StandardScaler()
X_train = scaler.fit_transform(attr_train)
X_test = scaler.transform(attr_test)

# Convert labels to one-hot encoding
encoder = LabelEncoder()
y_train = to_categorical(encoder.fit_transform(target_train))
y_test = to_categorical(encoder.transform(target_test))
# Build the neural network model
model = Sequential()
model.add(Dense(8, input_dim=9, activation='softmax'))
model.add(Dense(2, activation='softmax'))
# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.2, verbose=2)
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
#Ada Boost
# Split the data into training and testing sets
attr_train, attr_test, target_train, target_test = train_test_split(attr, target, test_size=0.2, random_state=7)

# Create AdaBoost classifier with Decision Tree as base estimator
estimator = DecisionTreeClassifier(max_depth=11)
ada_boost_classifier = AdaBoostClassifier(estimator=estimator, n_estimators=110, random_state=999)

# Fit the model
ada_boost_classifier.fit(attr_train, target_train)

# Make predictions
predictions = ada_boost_classifier.predict(attr_test)
# Evaluate the accuracy
accuracy = accuracy_score(target_test, predictions)
print(f'Accuracy: {accuracy}')
print()
cm =confusion_matrix(target_test, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="g", cmap="Purples", cbar=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
print()
print('Classification Report')
print(classification_report(target_test, predictions))
#Plot_tree
plt.figure(figsize=(50,30), dpi=250)
plot_tree(model,fontsize=20,filled=True,feature_names=attr.columns)
plt.show()
#RF Printout
# Feature importance for Mean Decrease Accuracy
mean_decrease_accuracy = model.feature_importances_
print(f"Mean Decrease Accuracy{ mean_decrease_accuracy}")
# # Feature importance for Mean Decrease GINI
# # Note: GINI importance is specific to decision trees and random forests
gini_importance = model.feature_importances_ * model.estimators_[0].tree_.impurity[0]
print(f"\nMean Decrease GINI= {gini_importance}")
plt.figure(figsize=(10, 6))
plt.barh(range(len(mean_decrease_accuracy)), mean_decrease_accuracy, align='center')
plt.yticks(range(len(mean_decrease_accuracy)), attr.columns)
plt.xlabel('Mean Decrease Accuracy')
plt.title('Feature Importance - Mean Decrease Accuracy')
plt.show()
plt.figure(figsize=(10, 6))
plt.barh(range(len(gini_importance)), gini_importance, align='center')
plt.yticks(range(len(gini_importance)), attr.columns)
plt.xlabel('Mean Decrease GINI')
plt.title('Feature Importance - Mean Decrease GINI')
plt.show()
feature_scores = pd.Series(model.feature_importances_, index=attr_train.columns).sort_values(ascending=False)
print(feature_scores)
df = pd.read_csv(csv_file_path_clean_num)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder,StandardScaler
X = df.drop('win_status', axis=1)
y = df['win_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
# Encoding the target variable
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)
# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# SVM model
svm_model = SVC(random_state=42)
svm_model.fit(X_train_scaled, y_train_encoded)
# ANN model
ann_model = MLPClassifier(random_state=42)
ann_model.fit(X_train_scaled, y_train_encoded)
# Predictions
svm_predictions = svm_model.predict(X_test_scaled)
ann_predictions = ann_model.predict(X_test_scaled)
# Evaluation
svm_accuracy = accuracy_score(y_test_encoded, svm_predictions)
ann_accuracy = accuracy_score(y_test_encoded, ann_predictions)
svm_report = classification_report(y_test_encoded, svm_predictions)
ann_report = classification_report(y_test_encoded, ann_predictions)

(svm_accuracy, ann_accuracy), (svm_report, ann_report)