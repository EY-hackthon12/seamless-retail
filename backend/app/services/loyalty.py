def compute_offer(points: int, subtotal: float) -> dict:
    redeem_value = min(points // 100, 5)  # $5 max in demo
    return {"redeem_value": redeem_value, "new_total": max(subtotal - redeem_value, 0)}
