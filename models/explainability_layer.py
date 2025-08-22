import pandas as pd
import numpy as np
import shap
from models.scoring_engine import CreditScoringModel


def generate_explanation(model: CreditScoringModel, input_features: pd.DataFrame, latest_events: list):
    """Generates a comprehensive, SHAP-based explanation for a given score."""

    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(input_features[model.features])

    # --- FINAL FIX: Average the SHAP values to get a single importance score per feature ---
    if isinstance(shap_values, list) and len(shap_values) > 1:
        # For a binary classifier, take the mean of the SHAP values for class 1 (good credit)
        # This gives a single, non-ambiguous value for the chart.
        shap_values_to_use = np.mean(shap_values[1], axis=0)
    else:
        # This handles the case where the training data only had one label.
        shap_values_to_use = np.abs(shap_values[0])

    feature_contributions = dict(zip(model.features, shap_values_to_use.tolist()))

    sorted_contributions = sorted(feature_contributions.items(), key=lambda item: item[1], reverse=True)
    main_driver = sorted_contributions[0][0] if sorted_contributions else "N/A"

    event_summary = []
    for event in latest_events:
        if event['sentiment'] == 'positive':
            impact = 'positive'
        elif event['sentiment'] == 'negative':
            impact = 'negative'
        else:
            impact = 'neutral'

        event_summary.append({
            'headline': event['headline'],
            'impact': impact,
            'reason': f"Detected a '{impact}' sentiment news headline."
        })

    latest_event_headline = latest_events[0]['headline'] if latest_events else "recent market events"

    summary_text = (
        f"The creditworthiness score is primarily driven by the **{main_driver.replace('_', ' ').title()}** feature. "
        f"Recent market events, such as **'{latest_event_headline}'**, are also influencing the score."
    )

    return {
        'feature_contributions': feature_contributions,
        'event_summary': event_summary,
        'summary': summary_text
    }