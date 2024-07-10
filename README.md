# Honkai Star Rail AI Team Recommender

Welcome to the Honkai Star Rail AI Team Recommender project! This project uses machine learning to help Honkai Star Rail players create optimal teams based on their available characters.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [How to Use](#how-to-use)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Honkai Star Rail AI Team Recommender is a system that leverages machine learning to recommend the best team combinations in the game Honkai Star Rail. By utilizing character data and existing team statistics, this system can help players make better decisions in composing their teams.

## Features

- Team recommendations based on the player's available characters
- Team statistics analysis including element and path diversity
- Team score prediction using a machine learning model
- Support for character and team data updates

## How to Use

1. Run the `Inference.py` script
2. Enter the list of characters you own when prompted
3. Input 1-3 initial characters for your team
4. The system will provide the best team recommendations based on your input

Example usage:

```
$ python Inference.py

Do you want to use all characters or a custom character list? (Y/N): N
Enter your available characters (comma-separated): Seele,Bronya,Pela,Natasha,Dan Heng
Enter 1 to 3 initial characters (comma-separated): Seele
Do you want to see detailed logging? (Y/N): Y

Team Recommendations:
1. ['Seele', 'Bronya', 'Pela', 'Natasha'] (Score: 0.9234)
2. ['Seele', 'Bronya', 'Dan Heng', 'Natasha'] (Score: 0.8976)
3. ['Seele', 'Pela', 'Dan Heng', 'Natasha'] (Score: 0.8543)
4. ['Seele', 'Bronya', 'Pela', 'Dan Heng'] (Score: 0.8321)
```

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/honkai-star-rail-ai-team-recommender.git
   ```
2. Navigate to the project directory:
   ```
   cd honkai-star-rail-ai-team-recommender
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Project Structure

- `Inference.py`: Main script for running team recommendations
- `Train.py`: Script for training the machine learning model
- `Dataset_Preprocessing.py`: Functions for dataset preprocessing
- `characters.csv`: Dataset of Honkai Star Rail characters
- `teams.csv`: Dataset of team combinations with scores
- `team_recommender_model.joblib`: Trained machine learning model

## Contributing

We greatly appreciate contributions from the community! If you'd like to contribute to this project, please follow these steps:

1. Fork this repository
2. Create a new feature branch (`git checkout -b cool-new-feature`)
3. Commit your changes (`git commit -am 'Add a cool new feature'`)
4. Push to the branch (`git push origin cool-new-feature`)
5. Create a new Pull Request

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
