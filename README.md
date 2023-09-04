# Text Classification with Various Machine Learning Models

This repository contains Python code for text classification using various machine learning models. The code reads SMS data from a CSV file, preprocesses the text data, trains different classifiers, evaluates their performance, and saves the trained models for future use.

## Prerequisites

Before running the code, ensure you have the following dependencies installed:

- Python 3.x
- NumPy
- Pandas
- NLTK (Natural Language Toolkit)
- Scikit-learn (sklearn)
- Joblib

You can install these dependencies using pip:

```
pip install numpy pandas nltk scikit-learn joblib
```

## Getting Started

1. Clone this repository to your local machine or download the code files.

2. Place the CSV file containing your SMS data in the same directory as the code file. Make sure to specify the correct file path in the following line of code:

   ```
   df = pd.read_csv('SMS_Dataset.csv', encoding='ISO-8859-1')
   ```

3. Run the code using a Python interpreter. This code performs the following steps:

   - Reads SMS data from the CSV file.
   - Downloads NLTK resources for tokenization and stopwords.
   - Preprocesses the text data by tokenizing, converting to lowercase, and removing stopwords.
   - Splits the data into training and testing sets.
   - Trains several machine learning classifiers, including Naive Bayes, Random Forest, Support Vector Machine (SVM), AdaBoost, Gradient Boosting, and Logistic Regression.
   - Evaluates the performance of each classifier using accuracy and classification reports.
   - Saves each trained classifier as a joblib (.pkl) file for future use.

## Customization

You can customize this code for your specific text classification task by doing the following:

- Replace `'SMS_Dataset.csv'` with the path to your own CSV file containing text data.

- Adjust the list of classifiers and their parameters in the `classifiers` list to experiment with different machine learning models and configurations.

- Modify the preprocessing steps in the `processText` function if you have specific text cleaning requirements.

## Usage

After running the code, you can use the trained classifier models (saved as `.pkl` files) for text classification tasks. Simply load the desired model using the `joblib.load` function and then use it to make predictions on new text data.

```
import joblib

# Load a trained model
model = joblib.load('Naive Bayes_model.pkl')

# Make predictions on new text data
new_text = ["Your new text goes here."]
predictions = model.predict(new_text)
print(predictions)
```

## License

This code is provided under the MIT License. Feel free to use and modify it for your own projects.
