from scam_job_detector.ML_logic.model import load_model, load_preprocessor
import shap
import os
import pandas as pd

    # ====================================================== #
    # SHAPLEY VALUES COMPUTATION
    # ====================================================== #

def shapley(X_new_preprocessed):

    # Load model and preprocessor
    model = load_model()
    preprocessor = load_preprocessor()

    # Load cleaned dataset
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    clean_data_path = os.path.join(base_path, "raw_data", "data_cleaned.csv")
    df = pd.read_csv(clean_data_path)
    print("✅ Clean data loaded")

    # function to make predictions
    def model_predict(X_new_preprocessed):
        return model.predict_proba(X_new_preprocessed)[:, 1]

    # Create 20-sample background dataset for SHAP
    background_df = (
        df
        .drop(columns=["fraudulent"], errors="ignore")
        .sample(20)
    )

    X_background = preprocessor.transform(background_df)
    explainer = shap.KernelExplainer(model_predict,X_background)
    shap_values = explainer.shap_values(X_new_preprocessed)

    #return output in Dataframe format
    feature_names = preprocessor.get_feature_names_out()
    len(feature_names), shap_values.shape[1]

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_values[0]
    })

    shap_df["abs_value"] = shap_df["shap_value"].abs()

    top_features = shap_df.sort_values(
        "abs_value", ascending=False
    ).head(20)
    top_features[["feature", "shap_value", "abs_value"]]

    #apply mask to separate text, binary and country features
    text_mask = shap_df.feature.str.contains("tfidfvectorizer__")
    binary_mask = shap_df.feature.str.contains("has_company_logo")
    country_mask = shap_df.feature.str.contains("country_")

    text_df = shap_df[text_mask]
    text_df = text_df[text_df.abs_value>0]
    binary_df = shap_df[binary_mask]
    country_df = shap_df[country_mask]

    # extract text from features names
    def extract_word(feature):
        return feature.split("tfidfvectorizer__")[-1]
    def extract_country(feature):
        return feature.split("country_")[-1]
    def extract_logo(feature):
        return feature.split("has_company_logo_")[-1]
    
    # Apply function to text_df
    text_df["word"] = text_df["feature"].apply(extract_word)
    binary_df["feature"] = binary_df["feature"].apply(extract_logo)
    country_df["feature"] = country_df["feature"].apply(extract_country)

    binary_mask = shap_df.feature.str.contains("has_company_logo")
    country_mask = shap_df.feature.str.contains("country_")

    shap_features_text = text_df["word"].tolist()
    shap_text_values =text_df["shap_value"].tolist()
    shap_features_binary = binary_df["feature"].tolist()
    shap_values_binary= binary_df["shap_value"].tolist()
    shap_features_country = country_df["feature"].tolist()
    shap_values_country= country_df["shap_value"].tolist()

    return shap_features_text, shap_text_values, shap_features_binary ,shap_values_binary, shap_features_country, shap_values_country