import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

FILE_PATH = f'data/{"emotion_data.csv"}'
class_weights = [0.3276, 0.2920, 0.1395, 0.1236, 0.0765, 0.0410]

# Loads data from a CSV file
# Our file path is "Emotion_data.csv", but it is passed in as a parameter
# returns the loaded data frame
def load_data(file_path):
    
    return pd.read_csv(file_path)


# trains a sentiment analysis model
# parameters: 
#   train data and its corresponding labels
#   as well as class weights, this is to balance out the disproportionate amount
#   of emotions in the dataset

# returns the trained model
def train_model(train_data, train_labels, class_weights=None):

    emotion_weights = dict(zip(['happy', 'sadness', 'anger', 'fear', 'love', 'surprise'], class_weights))

    # Create a pipeline with a text vectorizer, using MNB as the training algorithm
    model = make_pipeline(CountVectorizer(), MultinomialNB(class_prior=class_weights))

    # Fit the model with its train data
    model.fit(train_data, train_labels)

    return model


# Predicts percentages of each emotion category
# Parameters - model and user input text
# All percentages should sum up to 1
# returns a dictionary of all the percentages
def predict_emotion_percentages(model, text):
   
    # Make probability predictions of the text using the model
    predictions = model.predict_proba([text])

    # Extracts emotions from the model
    emotion_classes = model.classes_

    # Create a dictionary to store emotion percentages
    emotion_percentages = {emotion: percentage for emotion, percentage in zip(emotion_classes, predictions[0])}

    # Display emotion percentages
    print("\nEmotion Percentages:")
    for emotion, percentage in emotion_percentages.items():
        print(f"{emotion.capitalize()}: {percentage:.2%}")

    return emotion_percentages


# Using emotion percentage distribution, calculates a blended color of each emotion
# each emotion is mapped to a specific color, and RGB values are used to blend each color
# Parameter: the dictionary of emotion percentage distribution
# returns the hex number of the derived collor
def derive_color(emotion_percentages):

    # maps the emotion categories to specific colors
    # colors were determined with research
    # note that certain colors (green) are ommitted because of aesthetic purposes
    emotion_colors = {
        'happy':    '#FF8200',   # Vibrant Orange
        'sadness':  '#001A72',   # Deep Blue
        'anger':    '#C8102E',   # Bright Red
        'fear':     '#5F259F',   # Blue Purple
        'love':     '#F5B6CD',   # Light Pink 
        'surprise': '#FFCD00',   # Bright Yellow 
    }

    weighted_rgb = [0, 0, 0]

    # Blends colors by converting hex colors associated with emotions to RGB values 
    # and then multiplying these values by the percentage associated with it.
    # Sums up the RGB Values
    for emotion, percentage in emotion_percentages.items():
        color = [int(emotion_colors[emotion][i:i + 2], 16) for i in (1, 3, 5)]
        weighted_rgb = [weighted_rgb[i] + percentage * color[i] for i in range(3)]

    # Convert the weighted RGB components to hex number
    derived_color = "#{:02X}{:02X}{:02X}".format(*[int(component) for component in weighted_rgb])
    
    # calls function to display the color swatch
    display_color(derived_color)

    # return derived_color


# prints out the color swatch and also the hex nummber associated with the color
# parameter: derived hex color 
def display_color(color):
    print(f"Derived Color: \x1b[48;2;{int(color[1:3], 16)};{int(color[3:5], 16)};{int(color[5:], 16)}m{' ' * 10}\x1b[0m")
    print("Hex format: " + color)


# main program
def main():

    print("This program is designed to take some text and generate a color for you based off of emotion analysis!")

    # Loads dataset
    df = load_data(FILE_PATH)

    # Builds and Trains model
    model = train_model(df['Text'], df['Emotion'], class_weights)

    while True:

        # Asks for user input
        user_input = input("\nEnter some text (or type exit to end): ")

        # exits program if no more
        if user_input.lower() == 'exit':
            print("Exiting the program.")
            break

    # Calculates emotion percentages
        emotion_percentages = predict_emotion_percentages(model, user_input)

    # Generates the color
        derive_color(emotion_percentages)


if __name__ == "__main__":
    main()