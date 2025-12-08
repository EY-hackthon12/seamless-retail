import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

def generate_retail_data(num_days=365):
    """
    Generates synthetic retail data for a single store.
    Features: Date, DayOfWeek, IsHoliday, Promo, Footfall, InventoryLevel, Rainfall_mm, Sales
    """
    base_date = datetime(2024, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(num_days)]
    
    data = []
    
    for date in dates:
        # 1. Basic Features
        day_of_week = date.weekday() # 0=Monday, 6=Sunday
        is_weekend = 1 if day_of_week >= 5 else 0
        month = date.month
        
        # 2. Seasonality (Sine wave for yearly cycle)
        seasonality = np.sin(2 * np.pi * (date.timetuple().tm_yday / 365.0))
        
        # 3. Random Events
        is_holiday = 1 if random.random() < 0.05 else 0 # 5% chance of holiday
        promo_active = 1 if random.random() < 0.2 else 0 # 20% chance of promo
        
        # 4. Environmental
        rainfall = max(0, np.random.normal(5, 10)) if random.random() < 0.3 else 0
        
        # 5. Operations
        # Footfall depends on Weekend, Holiday, Promo, Seasonality, and Rain (negative impact)
        # Base footfall around 500
        footfall_mu = 500 + (is_weekend * 200) + (is_holiday * 100) + (promo_active * 150) + (seasonality * 50) - (rainfall * 5)
        footfall = int(max(50, np.random.normal(footfall_mu, 50)))
        
        inventory_level = int(np.random.normal(1000, 100)) # Random inventory fluctuation
        
        # 6. Target: Sales
        # Sales depend on Footfall, Promo, and random noise
        # Conversion rate approx 10-20%
        conversion_rate = 0.15 + (promo_active * 0.05)
        sales_mu = footfall * conversion_rate * np.random.uniform(20, 30) # Avg basket size
        
        # Constraints
        if inventory_level < 100:
             sales_mu *= 0.5 # Stockout impact
             
        sales = max(0, int(np.random.normal(sales_mu, sales_mu * 0.1)))
        
        data.append({
            "Date": date.strftime("%Y-%m-%d"),
            "DayOfWeek": day_of_week,
            "IsWeekend": is_weekend,
            "IsHoliday": is_holiday,
            "Promo": promo_active,
            "Rainfall": round(rainfall, 2),
            "Footfall": footfall,
            "Inventory": inventory_level,
            "Sales": sales
        })
        
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    print("--> Generating synthetic retail data...")
    df = generate_retail_data(num_days=1000)
    
    output_path = os.path.join(os.path.dirname(__file__), "retail_data.csv")
    df.to_csv(output_path, index=False)
    print(f"--> Data saved to {output_path}")
    print(df.head())
