import pandas as pd

# from google.cloud import bigquery
# from colorama import Fore, Style
# from pathlib import Path

# from taxifare.params import *

def preprocessing(sentence):

    stop_words = set(stopwords.words('english'))

    # remove punctuation
    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, '')

    # set to lowercase
    sentence = sentence.lower()

    # remove numbers
    for char in string.digits:
        sentence = ''.join(char for char in sentence if not char.isdigit())

    # tokenize
    tokens = word_tokenize(sentence)

    # removing stop words
    tokens = [word for word in tokens if word not in stop_words]

    # lemmatize
    tokens = [WordNetLemmatizer().lemmatize(word, pos='v') for word in tokens]

    return ' '.join(tokens)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by
    - Creating new features for columns with missing values above >30% as binary features: missing = 0, not missing = 1
    - Cleaning text data by removing stopwords, digits, lamatizing, etc.
    - 
    """
    df = read_csv('raw_data/fake_job_postings.csv')
    print('dataset loaded')

    # Creating binary columns for missing values:
    df['department_binary'] = df['department'].map(lambda x: 0 if pd.isna(x) else 1)
    
    df['salary_range_binary'] = df['salary_range'].map(lambda x: 0 if pd.isna(x) else 1)
    
    # Clean text data
    cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']

    df = df.copy()

    for col in cols:
        df[col] = df[col].fillna('missing value')

    for col in cols:
        df[col] = df[col].apply(preprocessing)
    
    # extracting country ID
    df['country'] = df['location'].astype(str).apply(lambda x: x.split(',')[0])

    # dropping columns
    df.drop(columns=['salary_range', 'department', 'location', 'job_id'], inplace=True)
    
    # store data
    df.to_csv('raw_data/data_cleaned.csv', index=False)

    print("✅ data cleaned")

    return df
