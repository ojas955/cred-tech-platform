import pandas as pd
from models.scoring_engine import CreditScoringModel

def generate_explanation(model: CreditScoringModel, input_features: pd.DataFrame, latest_events: list):
    """Generates a comprehensive explanation for a given score."""

    # 1. Feature contribution
    feature_contributions = model.get_feature_importance()

    # 2. Latest events & their impact (from unstructured data)
    event_summary = []
    for event in latest_events:
        impact = "positive" if event['sentiment'] == "positive" else "negative"
        event_summary.append({
            'headline': event['headline'],
            'impact': impact,
            'reason': f"Detected a '{impact}' sentiment news headline related to an event."
        })

    # 3. Plain-language summary
    input_data = input_features.to_dict('records')[0]

    # Get the top contributing feature from the model
    main_driver = max(feature_contributions, key=feature_contributions.get)

    # Get the latest event for a dynamic summary
    latest_event_headline = latest_events[0]['headline'] if latest_events else "recent market events"

    summary_text = (
        f"The creditworthiness score is primarily driven by changes in key financial ratios, "
        f"with a significant contribution from the **{main_driver}** feature. "
        f"Additionally, recent market events, such as **'{latest_event_headline}'** "
        f"are influencing the score."
    )

    return {
        'feature_contributions': feature_contributions,
        'event_summary': event_summary,
        'summary': summary_text
    }
