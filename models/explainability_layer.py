from models.scoring_engine import CreditScoringModel

def generate_explanation(model: CreditScoringModel, input_features, latest_events):
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
            'reason': f"Detected a '{impact}' sentiment news headline related to a '{event['event_type']}' event."
        })

    # 3. Plain-language summary
    summary_text = (
        f"The creditworthiness score is primarily driven by changes in key financial ratios, "
        f"with a significant contribution from the debt-to-equity ratio. "
        f"Additionally, recent market events, such as '{latest_events[0]['headline']}' "
        f"are influencing the score."
    )

    return {
        'feature_contributions': feature_contributions,
        'event_summary': event_summary,
        'summary': summary_text
    }