from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
app = FastAPI()
from pydantic import BaseModel
import shap
import joblib


rf_grid = joblib.load('rf_model.pkl')
rf_model = rf_grid.best_estimator_
preprocessor = joblib.load('preprocessor.pkl')
iso_forest = joblib.load('new_iso_forest.pkl')
explainer = shap.TreeExplainer(rf_model)

app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_headers=['*'],allow_methods=['*'])


class UserInputs(BaseModel):
    District: str
    Water_Source: str
    pH: float
    Hardness_mgL: float
    TDS_mgL: float
    Chloramines_mgL: float
    Sulfate_mgL: float
    Conductivity_uScm: float
    Organic_Carbon_mgL: float
    Trihalomethanes_ugL: float
    Turbidity_NTU: float


@app.get('/')
def sample_message():
    return {
        'message' : 'Use The Predict Endpoint to make Predictions'
    }


@app.post('/predict')
def predict_input(data:UserInputs):
    input_df = pd.DataFrame([data.model_dump()])
    processed_df = preprocessor.transform(input_df)
    anomaly = int(iso_forest.predict(processed_df)[0])
    prediction = int(rf_model.predict(processed_df)[0])
    shap_vals = explainer.shap_values(processed_df)
    if isinstance(shap_vals,list):
        class1 = shap_vals[1][0]
    else:
        class1 = shap_vals[0,:,1]
    raw_names = preprocessor.get_feature_names_out()
    impact_dictionary = {}
    for i in range(len(raw_names)):
        clean_names = raw_names[i].replace('num__','').replace('cat__','')
        impact_dictionary[clean_names] = round(float(class1[i]),4)

    if prediction == 0:
        worst_feature = min(impact_dictionary,key=impact_dictionary.get)
        msg = f"Unsafe: Biggest reason is {worst_feature} (impact: {impact_dictionary[worst_feature]})"
    elif anomaly==-1:
        msg = "Potential False Alarm: Predicted Safe, but chemical pattern is unusual."
    else:
        best_feature = max(impact_dictionary,key = impact_dictionary.get)            
        msg = f"Safe & Normal: Strongly supported by {best_feature} (impact: +{impact_dictionary[best_feature]})"

    return{
        'is_anomaly' : anomaly,
        'Potability' : prediction,
        'Message' : msg,
        'Explaination' : impact_dictionary
    }    