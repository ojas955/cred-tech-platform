from flask import Flask, render_template
import logging
from data_pipeline import run_pipeline  # Import the pipeline function
from models.scoring_engine import CreditScoringModel

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# --- GLOBAL VARIABLES ---
# Load the model and define companies when the app starts
COMPANIES_TO_TRACK = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
MODEL_FILE_PATH = 'credit_scoring_model.joblib'
SCORING_MODEL = CreditScoringModel.load_model(MODEL_FILE_PATH)


@app.route('/')
def dashboard():
    """
    Renders the main dashboard page.
    It runs the pipeline for all tracked companies and displays the results.
    """
    logging.info("Dashboard requested. Running pipeline for all companies...")
    all_results = []
    if SCORING_MODEL:
        for ticker in COMPANIES_TO_TRACK:
            result = run_pipeline(ticker, SCORING_MODEL)
            if result:
                all_results.append(result)
    else:
        logging.error("Dashboard cannot be loaded because the model is not available.")
        return "Error: Model not found. Please train the model first by running train.py", 500

    return render_template('index.html', results=all_results)


if __name__ == '__main__':
    # Note: For development, use `flask run`. For production, use a proper WSGI server.
    app.run(debug=True, port=5001)