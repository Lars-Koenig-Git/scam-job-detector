from scam_job_detector.ML_logic.model import load_model, load_preprocessor
import shap
import os
import pandas as pd
from wordcloud import WordCloud

    # ====================================================== #
    # 3️⃣ SHAPLEY VALUES COMPUTATION
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
        .sample(20, random_state=42)
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

    shap_features = shap_df["feature"].tolist()
    shap_values_list =shap_df["shap_value"].tolist()

    return shap_features, shap_values_list

def generate_wordcloud(shap_features, shap_values_list):
    # Create a dictionary of feature names and their corresponding SHAP values
    shap_dict = dict(zip(shap_features, shap_values_list))

    # Generate word cloud
    wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(shap_dict)

    return wc
