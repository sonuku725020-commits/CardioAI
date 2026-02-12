
import joblib
import numpy as np
import pandas as pd
import json

# Load saved model and scaler
def load_model(model_dir='heart_disease_model'):
    """Load the saved model and preprocessing components"""
    model = joblib.load(f'{model_dir}/gradient_boosting_model.pkl')
    scaler = joblib.load(f'{model_dir}/scaler.pkl')
    
    # Load feature info
    with open(f'{model_dir}/feature_info.json', 'r') as f:
        feature_info = json.load(f)
    
    return model, scaler, feature_info

def predict_heart_disease(patient_data, model_dir='heart_disease_model'):
    """
    Predict heart disease for a patient
    
    Parameters:
    -----------
    patient_data : dict or DataFrame
        Patient features matching the training data format
        
    Returns:
    --------
    dict : Prediction results with probabilities and interpretation
    """
    # Load model, scaler, and feature info
    model, scaler, feature_info = load_model(model_dir)
    
    # Expected features (excluding patientid and target)
    expected_features = feature_info['feature_names']
    
    # Convert to DataFrame if dict
    if isinstance(patient_data, dict):
        patient_df = pd.DataFrame([patient_data])
    else:
        patient_df = patient_data
    
    # Ensure all features are present
    missing_features = set(expected_features) - set(patient_df.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Select and order features
    patient_df = patient_df[expected_features]
    
    # Scale features
    patient_scaled = scaler.transform(patient_df)
    
    # Make prediction
    prediction = model.predict(patient_scaled)[0]
    probabilities = model.predict_proba(patient_scaled)[0]
    
    # Determine risk level
    disease_prob = probabilities[1]
    if disease_prob >= 0.8:
        risk_level = 'Very High'
        risk_color = 'red'
    elif disease_prob >= 0.6:
        risk_level = 'High'
        risk_color = 'orange'
    elif disease_prob >= 0.4:
        risk_level = 'Medium'
        risk_color = 'yellow'
    elif disease_prob >= 0.2:
        risk_level = 'Low'
        risk_color = 'lightgreen'
    else:
        risk_level = 'Very Low'
        risk_color = 'green'
    
    # Get feature importance for this prediction
    top_features = feature_info['top_5_features']
    patient_feature_values = {}
    for feature in top_features:
        patient_feature_values[feature] = float(patient_df[feature].iloc[0])
    
    # Prepare comprehensive results
    result = {
        'prediction': int(prediction),
        'prediction_label': 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease',
        'probability_no_disease': float(probabilities[0]),
        'probability_disease': float(probabilities[1]),
        'confidence': float(max(probabilities)),
        'risk_level': risk_level,
        'risk_color': risk_color,
        'recommendation': get_recommendation(disease_prob),
        'top_contributing_features': patient_feature_values,
        'interpretation': get_interpretation(disease_prob)
    }
    
    return result

def get_recommendation(disease_probability):
    """Generate recommendations based on disease probability"""
    if disease_probability >= 0.8:
        return "URGENT: Immediate medical consultation recommended. High risk of heart disease detected."
    elif disease_probability >= 0.6:
        return "Schedule a medical consultation soon. Elevated risk of heart disease."
    elif disease_probability >= 0.4:
        return "Consider scheduling a check-up with your physician for further evaluation."
    elif disease_probability >= 0.2:
        return "Continue monitoring. Maintain healthy lifestyle habits."
    else:
        return "Low risk detected. Continue with regular health check-ups and healthy lifestyle."

def get_interpretation(disease_probability):
    """Provide interpretation of the results"""
    percentage = disease_probability * 100
    return f"Based on the provided medical parameters, there is a {percentage:.1f}% probability of heart disease presence."

# Example usage
if __name__ == "__main__":
    # Example patient data
    example_patient = {
        'age': 55,
        'gender': 1,  # 1 for male, 0 for female
        'chestpain': 2,  # chest pain type (0-3)
        'restingBP': 140,
        'serumcholestrol': 260,
        'fastingbloodsugar': 0,  # 0 or 1
        'restingrelectro': 1,  # 0-2
        'maxheartrate': 150,
        'exerciseangia': 0,  # 0 or 1
        'oldpeak': 1.5,
        'slope': 1,  # 0-2
        'noofmajorvessels': 0  # 0-3
    }
    
    result = predict_heart_disease(example_patient)
    
    print("="*60)
    print("HEART DISEASE PREDICTION RESULTS")
    print("="*60)
    print(f"Prediction: {result['prediction_label']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"\nProbabilities:")
    print(f"  No Disease: {result['probability_no_disease']:.1%}")
    print(f"  Disease:    {result['probability_disease']:.1%}")
    print(f"\nRisk Level: {result['risk_level']}")
    print(f"\nInterpretation: {result['interpretation']}")
    print(f"\nRecommendation: {result['recommendation']}")
    print("="*60)
