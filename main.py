from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse # <-- Add this import
from pydantic import BaseModel
import joblib
import pandas as pd
import time
import io
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# 1. Initialize the App
app = FastAPI(title="Fraud Detection API")

# 2. Enable CORS (Crucial for connecting to your HTML frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (change this in production!)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    temperature = 0,
    groq_api_key = os.getenv("groq_api")
)


# 3. Load your trained model on startup
print("Loading model...")
model = joblib.load("fraud_detection_model.pkl")
print("Model loaded successfully!")

# 4. Define the expected incoming JSON structure
class TransactionInput(BaseModel):
    amount: float
    type_txn: str
    recency_hours: float
    txn_count_24hr: float
    hours_of_day: float
    is_dest_new: int
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float

# 5. Create the prediction endpoint
@app.post("/predict")
async def predict_fraud(transaction: TransactionInput):
    start_time = time.perf_counter()

    # 1. Initialize the dictionary with EVERY column the model expects
    # The order here doesn't matter yet, but the names must match exactly
    feature_dict = {
        'amount': transaction.amount,
        'recency_hours': transaction.recency_hours,
        'txn_count_24hr': transaction.txn_count_24hr,
        'is_dest_new': transaction.is_dest_new,
        'hours_of_day': transaction.hours_of_day,
        'oldbalanceOrg': transaction.oldbalanceOrg,
        'newbalanceOrig': transaction.newbalanceOrig,
        'oldbalanceDest': transaction.oldbalanceDest,
        'newbalanceDest': transaction.newbalanceDest,
        'type_CASH_IN': 0,
        'type_CASH_OUT': 0,
        'type_DEBIT': 0,
        'type_PAYMENT': 0,
        'type_TRANSFER': 0
    }

    # 2. Set the correct transaction type to 1
    type_key = f"type_{transaction.type_txn.upper()}"
    if type_key in feature_dict:
        feature_dict[type_key] = 1

    # 3. Convert to DataFrame
    input_df = pd.DataFrame([feature_dict])

    expected_order = [
        'amount', 'recency_hours', 'txn_count_24hr', 'is_dest_new', 
        'hours_of_day', 'oldbalanceOrg', 'newbalanceOrig', 
        'oldbalanceDest', 'newbalanceDest', 'type_CASH_IN', 
        'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'
    ]
    input_df = input_df[expected_order]

    # 5. Make the Prediction
    probability = float(model.predict_proba(input_df)[0][1])
    
    is_fraud = bool(probability > 0.55)

    end_time = time.perf_counter()
    processing_time_ms = round((end_time - start_time) * 1000, 2)

    summary = None

    if probability > 0.55:
        prompt = f"""You are a fraud detection analyst. Analyze this transaction and provide a clear, professional summary.

ANALYSIS RESULTS:
- Fraud Probability: {probability*100:.2f}%
- Decision: {"FRAUD DETECTED"}
- Risk Level: {"HIGH"}

TRANSACTION DETAILS:
- Amount: ${transaction.amount:,.2f}
- Type: {transaction.type_txn.upper()}
- Account Balance Before: ${transaction.oldbalanceOrg:,.2f}
- Account Balance After: ${transaction.newbalanceOrig:,.2f}
- Destination Balance Before: ${transaction.oldbalanceDest:,.2f}
- Destination Balance After: ${transaction.newbalanceDest:,.2f}
- New Destination: {"Yes" if transaction.is_dest_new == 1 else "No"}
- Hours Since Last Transaction: {transaction.recency_hours}
- Transactions in Last 24h: {transaction.txn_count_24hr}

Provide a 2-3 sentence professional summary explaining why this transaction is {"flagged as fraud"}. Focus on the key risk indicators."""

        try:
            response = llm.invoke(
                [
                    {"role": "system", "content": "You are a financial fraud detection expert."},
                    {"role": "user", "content": prompt}
                ]
            )
            summary = response.content.strip()

        except Exception as e:
            summary = f"Error generating summary: {str(e)}"

    response_data = {
        "is_fraud": is_fraud,
        "probability": probability,
        "processing_time_ms": processing_time_ms
    }

    if summary is not None:
        response_data["summary"] = summary

    return response_data

    
@app.post("/predict_batch")
async def predict_batch_fraud(file: UploadFile = File(...)):
    start_time = time.perf_counter()
    
    # Read the CSV content into a pandas DataFrame
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    # Clean up column names (stripping whitespace) just in case
    df.columns = df.columns.str.strip()

    expected_order = [
         'amount', 'recency_hours', 'txn_count_24hr', 'is_dest_new', 
         'hours_of_day', 'oldbalanceOrg', 'newbalanceOrig', 
         'oldbalanceDest', 'newbalanceDest', 'type_CASH_IN', 
         'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'
    ]

    input_df = pd.DataFrame()

    # Base common columns
    base_cols = [
        'amount', 'recency_hours', 'txn_count_24hr', 
        'hours_of_day', 'is_dest_new', 'oldbalanceOrg', 'newbalanceOrig', 
        'oldbalanceDest', 'newbalanceDest'
    ]
    
    # Check for missing base columns
    missing_base = [c for c in base_cols if c not in df.columns]
    if missing_base:
        return {"error": f"Missing required columns: {missing_base}"}

    for col in base_cols:
        input_df[col] = df[col]

    # Handle Transaction Type (either single 'type_txn' column or already one-hot encoded)
    if 'type_txn' in df.columns:
        txn_types = df['type_txn'].astype(str).str.upper().str.strip()
        input_df['type_CASH_IN'] = (txn_types == 'CASH_IN').astype(int)
        input_df['type_CASH_OUT'] = (txn_types == 'CASH_OUT').astype(int)
        input_df['type_DEBIT'] = (txn_types == 'DEBIT').astype(int)
        input_df['type_PAYMENT'] = (txn_types == 'PAYMENT').astype(int)
        input_df['type_TRANSFER'] = (txn_types == 'TRANSFER').astype(int)
    else:
        # Check if one-hot columns already exist in the CSV (like in test chunks)
        one_hot_cols = ['type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
        missing_one_hot = [c for c in one_hot_cols if c not in df.columns]
        
        if missing_one_hot:
            return {"error": f"Missing transaction type columns. Need either 'type_txn' or all of {one_hot_cols}"}
            
        for col in one_hot_cols:
            input_df[col] = df[col]

    # Order exactly as expected by the model
    input_df = input_df[expected_order]

    # Make batch predictions using predict_proba
    probabilities = model.predict_proba(input_df)[:, 1]
    
    # Format the results
    results = []
    for prob in probabilities:
        results.append({
            "is_fraud": bool(prob > 0.55),
            "probability": float(prob)
        })

    end_time = time.perf_counter()
    processing_time_ms = round((end_time - start_time) * 1000, 2)

    return {
        "results": results,
        "processing_time_ms": processing_time_ms
    }

@app.get("/")
async def serve_frontend():
    # Looks for page.html in the same directory
    html_file_path = os.path.join(os.path.dirname(__file__), "page.html")
    if os.path.exists(html_file_path):
        return FileResponse(html_file_path)
    return {"error": "Frontend page.html not found"}