import string
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import os

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by
    - Creating new features for columns with missing values above >30% as binary features: missing = 0, not missing = 1
    - Cleaning text data by removing stopwords, digits, lamatizing, etc.
    -
    """
    def preprocessing(sentence):

        stop_words = set(stopwords.words('english'))

        # remove punctuation
        for punctuation in string.punctuation:
            sentence = sentence.replace(punctuation, ' ')

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
    # text columns for TF-IDF Vectorizer
    text_columns = [
            'title',
            'company_profile',
            'description',
            'requirements',
            'benefits'
    ]
    df[text_columns] = df[text_columns].fillna("").astype(str)
    df["job_description"] = df[text_columns].agg(" ".join, axis=1).str.strip()

    # Clean text data
    df['job_description'] = df['job_description'].fillna('missing value')
    df['job_description'] = df['job_description'].apply(preprocessing)

    # extracting country ID
    df['country'] = df['location'].astype(str).apply(lambda x: x.split(',')[0])

    # reducing number of categories for the most important variables as shown by feature permutation.
    top10_idx = df["industry"].value_counts().head(10).index
    df["industry"] = df["industry"].where(df["industry"].isin(top10_idx), "Other")

    top10_idx = df["country"].value_counts().head(10).index
    df["country"] = df["country"].where(df["country"].isin(top10_idx), "Other")
    

    top10_idx = df["function"].value_counts().head(10).index
    df["function"] = df["function"].where(df["function"].isin(top10_idx), "Other")
    

    # Drop irrelevant columns
    df.drop(columns=['department',
                    'has_questions',
                     'required_experience',
                    'salary_range',
                     'telecommuting',
                     'required_education',
                     'location',
                     'title',
                     'company_profile',
                     'description',
                     'requirements',
                     'benefits',
                     'job_id'], inplace=True)

    print("✅ data cleaned")

    return df

def save_clean_data():
    """
    Clean raw data ONCE and save it to raw_data/data_cleaned.csv
    """
    raw_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'raw_data',
        'fake_job_postings.csv'
    )
    df_raw = pd.read_csv(raw_path)

    df_clean = clean_data(df_raw)

    save_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'raw_data',
        'data_cleaned.csv'
    )
    df_clean.to_csv(save_path, index=False)

    print(f"✅ Cleaned data saved at: {save_path}")

if __name__ == "__main__":
    save_clean_data()
