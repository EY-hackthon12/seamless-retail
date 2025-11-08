def process(amount: float, method: str = "card", token: str | None = None) -> dict:
    return {"status": "authorized", "amount": amount}
