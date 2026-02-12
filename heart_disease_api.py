"""
Simple FastAPI for Heart Disease Prediction
Run with: uvicorn heart_disease_api:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
import json
from pathlib import Path

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class PatientData(BaseModel):
    """Input model for patient data"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 55,
                "gender": 1,
                "chestpain": 2,
                "restingBP": 140,
                "serumcholestrol": 260,
                "fastingbloodsugar": 0,
                "restingrelectro": 1,
                "maxheartrate": 150,
                "exerciseangia": 0,
                "oldpeak": 1.5,
                "slope": 1,
                "noofmajorvessels": 0
            }
        }
    )
    
    age: int = Field(..., ge=1, le=120, description="Age in years")
    gender: int = Field(..., ge=0, le=1, description="0: Female, 1: Male")
    chestpain: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    restingBP: int = Field(..., ge=50, le=250, description="Resting blood pressure")
    serumcholestrol: int = Field(..., ge=100, le=600, description="Serum cholesterol")
    fastingbloodsugar: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120")
    restingrelectro: int = Field(..., ge=0, le=2, description="Resting ECG (0-2)")
    maxheartrate: int = Field(..., ge=50, le=250, description="Maximum heart rate")
    exerciseangia: int = Field(..., ge=0, le=1, description="Exercise induced angina")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise")
    noofmajorvessels: int = Field(..., ge=0, le=4, description="Number of major vessels")


class PredictionResponse(BaseModel):
    """Output model for prediction"""
    prediction: int
    prediction_label: str
    probability_no_disease: float
    probability_disease: float
    risk_level: str
    recommendation: str


# ============================================================================
# HEART DISEASE PREDICTOR
# ============================================================================

class HeartDiseasePredictor:
    def __init__(self, model_dir: str = "heart_disease_model"):
        """Initialize predictor with model files"""
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.feature_names = [
            'age', 'gender', 'chestpain', 'restingBP', 'serumcholestrol',
            'fastingbloodsugar', 'restingrelectro', 'maxheartrate',
            'exerciseangia', 'oldpeak', 'slope', 'noofmajorvessels'
        ]
        self.load_model()
    
    def load_model(self):
        """Load model and scaler"""
        try:
            self.model = joblib.load(self.model_dir / "gradient_boosting_model.pkl")
            self.scaler = joblib.load(self.model_dir / "scaler.pkl")
            print(f"[OK] Model loaded successfully")
        except Exception as e:
            print(f"[ERROR] Error loading model: {e}")
            raise
    
    def predict(self, patient_data: Dict) -> Dict:
        """Make prediction for a patient"""
        # Create DataFrame
        df = pd.DataFrame([patient_data])
        df = df[self.feature_names]
        
        # Scale features
        scaled_data = self.scaler.transform(df)
        
        # Predict
        prediction = self.model.predict(scaled_data)[0]
        probabilities = self.model.predict_proba(scaled_data)[0]
        
        # Determine risk level
        disease_prob = probabilities[1]
        if disease_prob >= 0.8:
            risk_level = "Very High"
            recommendation = "🚨 URGENT: Immediate medical consultation recommended!"
        elif disease_prob >= 0.6:
            risk_level = "High"
            recommendation = "⚠️ Schedule medical consultation soon."
        elif disease_prob >= 0.4:
            risk_level = "Medium"
            recommendation = "📋 Consider scheduling a check-up."
        elif disease_prob >= 0.2:
            risk_level = "Low"
            recommendation = "✅ Continue monitoring your health."
        else:
            risk_level = "Very Low"
            recommendation = "💚 Maintain healthy lifestyle!"
        
        return {
            "prediction": int(prediction),
            "prediction_label": "Heart Disease" if prediction == 1 else "No Heart Disease",
            "probability_no_disease": float(probabilities[0]),
            "probability_disease": float(probabilities[1]),
            "risk_level": risk_level,
            "recommendation": recommendation
        }


# ============================================================================
# FASTAPI APP
# ============================================================================

# Create FastAPI instance
app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predict heart disease risk using machine learning",
    version="1.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
predictor = HeartDiseasePredictor()


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
def home():
    """Welcome endpoint"""
    return {
        "message": "🫀 Heart Disease Prediction API",
        "documentation": "/docs",
        "health_check": "/health"
    }


@app.get("/health")
def health_check():
    """Check API and model status"""
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None,
        "scaler_loaded": predictor.scaler is not None
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_heart_disease(patient: PatientData):
    """
    Predict heart disease for a patient
    
    Returns:
    - prediction: 0 (No Disease) or 1 (Disease)
    - probabilities: Probability for each class
    - risk_level: Very Low, Low, Medium, High, Very High
    - recommendation: Medical recommendation
    """
    try:
        # Convert to dict
        patient_dict = patient.model_dump()
        
        # Make prediction
        result = predictor.predict(patient_dict)
        
        return PredictionResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(patients: List[PatientData]):
    """Predict heart disease for multiple patients"""
    try:
        results = []
        for patient in patients:
            patient_dict = patient.model_dump()
            result = predictor.predict(patient_dict)
            results.append(result)
        
        # Calculate summary
        total = len(results)
        disease_count = sum(1 for r in results if r["prediction"] == 1)
        
        return {
            "total_patients": total,
            "predictions": results,
            "summary": {
                "disease_detected": disease_count,
                "no_disease": total - disease_count,
                "disease_percentage": round(disease_count/total * 100, 2)
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
def model_info():
    """Get model information"""
    return {
        "model_type": "Gradient Boosting Classifier",
        "features": predictor.feature_names,
        "total_features": len(predictor.feature_names),
        "classes": ["No Disease", "Heart Disease"]
    }


@app.get("/example")
def get_example():
    """Get example request"""
    return {
        "description": "Example patient data",
        "patient_data": {
            "age": 55,
            "gender": 1,
            "chestpain": 2,
            "restingBP": 140,
            "serumcholestrol": 260,
            "fastingbloodsugar": 0,
            "restingrelectro": 1,
            "maxheartrate": 150,
            "exerciseangia": 0,
            "oldpeak": 1.5,
            "slope": 1,
            "noofmajorvessels": 0
        }
    }


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)