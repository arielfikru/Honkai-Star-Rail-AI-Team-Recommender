import joblib
from itertools import combinations
from Dataset_Preprocessing import load_character_dataset, preprocess_character_dataset, team_to_features, load_team_dataset
import pandas as pd
import logging
import sys

logging.basicConfig(level=logging.INFO)

def load_model():
    return joblib.load('team_recommender_model.joblib')

def get_team_recommendations(user_characters, initial_team, num_recommendations=4, verbose=False):
    regressor = load_model()
    df_characters = load_character_dataset()
    element_dict, path_dict, rarity_dict = preprocess_character_dataset(df_characters)
    
    # Load the original team dataset
    teams_df = load_team_dataset()
    
    available_characters = list(set(user_characters) - set(initial_team))
    slots_to_fill = 4 - len(initial_team)
    
    all_combinations = list(combinations(available_characters, slots_to_fill))
    recommendations = []

    feature_names = regressor.feature_names_in_
    
    for combo in all_combinations:
        team = initial_team + list(combo)  # Keep initial team order
        
        # Check if the team exists in the original dataset
        team_set = set(team)
        exact_match = teams_df[teams_df.apply(lambda row: set(row[['Character1', 'Character2', 'Character3', 'Character4']]) == team_set, axis=1)]
        
        if not exact_match.empty:
            score = exact_match['Score'].values[0]
            if verbose:
                logging.info(f"Exact match found for team {team} with score {score}")
        else:
            team_features = team_to_features(team, element_dict, path_dict, rarity_dict)
            features_df = pd.DataFrame([team_features], columns=feature_names)
            score = regressor.predict(features_df)[0]
            if verbose:
                logging.info(f"Predicted score for team {team}: {score}")
        
        recommendations.append((team, score, team_features))
    
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:num_recommendations]

def validate_characters(characters, all_valid_characters):
    invalid_characters = set(characters) - set(all_valid_characters)
    if invalid_characters:
        print(f"Error: The following characters are not valid: {', '.join(invalid_characters)}")
        print("Please check the character names and try again.")
        sys.exit(1)

if __name__ == "__main__":
    all_characters = "Acheron,Argenti,Arlan,Asta,Aventurine,Bailu,Black Swan,Blade,Boothill,Bronya,Clara,Dan Heng,Dr. Ratio,Firefly,Fu Xuan,Gallagher,Gepard,Guinafen,Hanya,Herta,Himeko,Hook,Huohuo,Imbibitor Lunae,Jade,Jing Yuan,Jingliu,Kafka,Luka,Luocha,Lynx,March 7th,Misha,Natasha,Pela,Qingque,Robin,Ruan Mei,Sampo,Seele,Serval,Silver Wolf,Sparkle,Sushang,Tingyun,Topaz,Trailblazer,Trailblazer(Fire),Trailblazer(Imaginary),Welt,Xueyi,Yanqing,Yukong"
    all_valid_characters = [char.strip() for char in all_characters.split(',')]

    use_all = input("Apakah kamu ingin menggunakan semua karakter atau Custom karakter list? (Y/N): ").strip().lower()
    
    if use_all == 'y':
        user_characters = all_valid_characters
    else:
        user_input = input("Enter your available characters (comma-separated): ")
        user_characters = [char.strip() for char in user_input.split(',')]
        validate_characters(user_characters, all_valid_characters)

    initial_input = input("Enter 1 to 3 initial characters (comma-separated): ")
    initial_team = [char.strip() for char in initial_input.split(',')]

    # Validate initial team characters
    validate_characters(initial_team, all_valid_characters)

    # Check if initial team characters are in user_characters
    invalid_initial = set(initial_team) - set(user_characters)
    if invalid_initial:
        print(f"Error: The following initial team characters are not in your available characters: {', '.join(invalid_initial)}")
        print("Please check your input and try again.")
        sys.exit(1)

    verbose_input = input("Do you want to see detailed logging? (Y/N): ").strip().lower()
    verbose = verbose_input == 'y'

    if len(initial_team) < 1 or len(initial_team) > 3:
        print("Error: Initial team must consist of 1 to 3 characters.")
        sys.exit(1)
    else:
        recommendations = get_team_recommendations(user_characters, initial_team, verbose=verbose)
        
        print("\nTeam Recommendations:")
        for i, (team, score, features) in enumerate(recommendations, 1):
            print(f"{i}. {team} (Score: {score:.4f})")
            
        print("\nDetailed feature information for the top recommendation:")
        top_team, top_score, top_features = recommendations[0]
        for feature, value in top_features.items():
            print(f"{feature}: {value}")

        print("\nModel feature importances:")
        feature_importances = load_model().feature_importances_
        feature_names = load_model().feature_names_in_
        for name, importance in sorted(zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True):
            print(f"{name}: {importance:.4f}")