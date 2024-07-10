import numpy as np
import pandas as pd

def load_character_dataset():
    return pd.read_csv('characters.csv')

def load_team_dataset():
    return pd.read_csv('teams.csv')

def preprocess_character_dataset(df_characters):
    element_dict = dict(zip(df_characters['Name'], df_characters['Element']))
    path_dict = dict(zip(df_characters['Name'], df_characters['Path']))
    rarity_dict = dict(zip(df_characters['Name'], df_characters['Rarity']))
    return element_dict, path_dict, rarity_dict

def team_to_features(team, element_dict, path_dict, rarity_dict):
    elements = [element_dict[char] for char in team]
    paths = [path_dict[char] for char in team]
    rarities = [rarity_dict[char] for char in team]
    
    features = {
        'avg_rarity': sum(rarities) / 4,
        'element_diversity': len(set(elements)),
        'path_diversity': len(set(paths)),
    }
    
    for element in set(element_dict.values()):
        features[f'count_{element}'] = elements.count(element)
    
    for path in set(path_dict.values()):
        features[f'count_{path}'] = paths.count(path)
    
    return features

def prepare_dataset():
    df_characters = load_character_dataset()
    element_dict, path_dict, rarity_dict = preprocess_character_dataset(df_characters)
    
    teams_df = load_team_dataset()
    
    X = []
    y = []
    
    for _, row in teams_df.iterrows():
        team = [row['Character1'], row['Character2'], row['Character3'], row['Character4']]
        features = team_to_features(team, element_dict, path_dict, rarity_dict)
        X.append(features)
        y.append(row['Score'])
    
    return pd.DataFrame(X), np.array(y), df_characters

if __name__ == "__main__":
    X, y, df_characters = prepare_dataset()
    print("Dataset shape:", X.shape)
    print("Labels shape:", y.shape)
    print("Sample features:")
    print(X.head())
    print("\nSample team score:")
    print(y[0])