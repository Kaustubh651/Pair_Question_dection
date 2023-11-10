# Pair_Question_dection


1. **Data Preprocessing:**
   - The script reads a CSV file ('train.csv') using the pandas library and creates a smaller random sample (`new_df`) of 30,000 rows.
   - A preprocessing function (`preprocess`) is defined to clean and standardize text data in the columns 'question1' and 'question2'. This function handles tasks such as converting to lowercase, replacing special characters, decontracting words, and removing HTML tags.

2. **Feature Engineering:**
   - The script adds several features to the dataset, including the length of questions, the number of words in each question, and common word statistics.
   - Advanced features such as token-based features, length-based features, and fuzzy features (using the fuzzywuzzy library) are also generated.
   - The script uses t-Distributed Stochastic Neighbor Embedding (t-SNE) for dimensionality reduction and visualization of the engineered features.

3. **Machine Learning Models:**
   - The dataset is split into features and labels for training machine learning models.
   - Two classifiers, RandomForestClassifier and XGBClassifier, are trained on the dataset to predict the 'is_duplicate' label.

4. **Evaluation and Visualization:**
   - The script evaluates the models using metrics such as accuracy and confusion matrices.
   - Visualization using seaborn and matplotlib is employed to explore the relationships between various features and the target variable.

5. **Deployment:**
   - The script defines functions (`query_point_creator`) to preprocess and create a feature vector for new input questions.
   - The trained RandomForestClassifier and CountVectorizer models are then saved using the pickle library for potential deployment.


### Project Overview:
The project appears to be a duplicate question pair detection system using a machine learning model. It seems to be implemented in Python and uses various libraries for natural language processing (NLP) and machine learning. The key components include:

1. **Libraries Used:**
   - `re`: Regular expressions library for text processing.
   - `BeautifulSoup`: Used for HTML parsing.
   - `distance`: Library for calculating string distances.
   - `fuzzywuzzy`: Library for fuzzy string matching.
   - `pickle`: Serialization library for saving and loading Python objects.
   - `numpy`: Library for numerical operations.
   - `nltk`: Natural Language Toolkit for text processing.
   - `streamlit`: A library for creating web applications.

2. **Functions:**
   - `test_common_words` and `test_total_words`: Functions to calculate the number of common and total words between two sentences.
   - `test_fetch_token_features`: Extracts features related to common words, stopwords, and tokens.
   - `test_fetch_length_features`: Computes features related to the length of sentences.
   - `test_fetch_fuzzy_features`: Utilizes fuzzy string matching to extract features.
   - `preprocess`: Preprocessing function for text data, including replacing special characters, handling numbers, and decontracting words.
   - `query_point_creator`: Creates a feature vector for a pair of input questions, combining basic, token, length, and fuzzy features.


3. **Streamlit Web Application:**
   - The code utilizes Streamlit to create a simple web interface for users to input two questions and find out whether they are duplicate or not.

### How to Run the Code:
1. Ensure you have all the required libraries installed. You can use `pip install -r requirements.txt` if you have a requirements file.
2. Load the trained model (`model.pkl`) and the CountVectorizer (`cv.pkl`) used for transforming text into numerical features.
3. Run the Streamlit application by executing the script "streamlit run app.py

### Suggestions:
- Ensure that all dependencies are installed and compatible with the provided code.
- Include appropriate error handling and user feedback in the Streamlit app.
- Consider providing more context about the dataset used for training the model.
- Include documentation and comments in the code for better readability and understanding.
- You may want to consider using virtual environments to manage dependencies.

This overview assumes that the model training and data preparation processes have been completed successfully, and the saved model files are available. If you have any specific questions or need further clarification on certain aspects of the code, feel free to ask!
