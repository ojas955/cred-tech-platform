CredTech: Real-Time Explainable Credit Intelligence Platform
1. Problem Statement
In the global credit markets, traditional credit ratings are often infrequent [cite: 8][cite_start], based on opaque methodologies [cite: 9][cite_start], and lag behind real-world events[cite: 10]. [cite_start]Our project addresses this by building a transparent, real-time, and explainable credit intelligence platform [cite: 17] powered by public data and machine learning.
2. Our Solution
Our platform provides a real-time creditworthiness score and, crucially, explains *why* the score was assigned[cite: 20]. [cite_start]It continuously processes multi-source data [cite: 18][cite_start], generates scores that react faster than traditional ratings [cite: 19][cite_start], and presents results through a web dashboard[cite: 22].
Real-Time Data Ingestion:The system fetches the latest financial data from Yahoo Finance and real-time news headlines from the NewsAPI[cite: 27, 29].
Adaptive Scoring Engine: We used a Random Forest model, which is a "black box" model, and integrated it with an explainability layer[cite: 38].
SHAP-Powered Explainability: We integrated the state-of-the-art SHAP library to provide clear, feature-level breakdowns for each score [cite: 42][cite_start], avoiding the use of LLMs for explanations as per the hackathon rules[cite: 45].
Interactive Dashboard: A web interface built with Flask displays the scores, key insights, and recent events, making it easy for an analyst to understand a company's financial health[cite: 85].

3. System Architecture
The application follows a modular and scalable architecture:
Backend:** Python, Flask.
Data Science: Pandas, Scikit-learn, SHAP, NLTK/VADER.
Data Sources: yfinance API, NewsAPI.
Frontend: HTML, Bootstrap, Chart.js.

The system pipeline is orchestrated by `data_pipeline.py`. It fetches structured financial data and unstructured news data, uses a pre-trained machine learning model to generate a score, and then an explainability layer provides a detailed, human-readable reason for the score[cite: 44]. The final result is sent to the Flask server, which renders the dashboard.

4. Live Demo

You can view the live, deployed version of the application here:

[https://cred-tech-platform-1.onrender.com/](https://cred-tech-platform-1.onrender.com/)

5. Key Trade-offs & Decisions

Proxy Target Variable:** Since no "true" credit score was available for training, we created a proxy target variable based on a company's future stock performance[cite: 95].
Sentiment Integration:** We chose to use real-time news sentiment as a post-model adjustment layer for maximum transparency and explainability, rather than as a direct model input, which would have required extensive historical sentiment data[cite: 56, 58].

 6. How to Run Locally

1.  Clone the repository:
    `git clone [Your Repository URL]`

2.  Navigate to the project directory:
    `cd [Your Project Directory]`

3.  Create a virtual environment:
    `python -m venv venv`
    `source venv/bin/activate`

4.  Install dependencies:
    `pip install -r requirements.txt`

5.  Create a `.env` file:**
    Create a new file named `.env` in the root directory and add your NewsAPI key:
    `NEWS_API_KEY=YOUR_API_KEY`

6.  Train the model:**
    `python train.py`
    This will create the `credit_scoring_model.joblib` file.

7.  Run the web application:**
    `flask run`

8.  Open your browser:**
    Open your web browser and navigate to `http://127.0.0.1:5000` to see the dashboard.
