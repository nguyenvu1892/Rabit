def detect_session(dt):

    hour = dt.hour

    # ===== ASIA =====
    if 0 <= hour < 7:
        return "asia"

    # ===== LONDON =====
    if 7 <= hour < 13:
        return "london"

    # ===== NEW YORK =====
    if 13 <= hour < 22:
        return "new_york"

    return "off_market"
