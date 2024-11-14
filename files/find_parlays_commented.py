from datetime import datetime
import math

import pandas as pd

# Variable assignment
# Variable assignment
# Variable assignment
df = pd.read_csv(f"total_odds{datetime.today().strftime('%y-%m-%d')}.csv", index_col=0)
# Variable assignment
df = df.sort_values("P[p_hat <= p_needed]")[df["P[p_hat <= p_needed]"] < 0.2]
df = df.reset_index()
# Variable assignment
# Loop starts here
# Variable assignment
# Variable assignment
# Variable assignment
# Variable assignment
# Variable assignment
# Loop starts here
# Conditional statement
# Variable assignment
# Variable assignment
# Variable assignment
# Variable assignment
# Variable assignment
# Loop starts here
# Conditional statement
# Variable assignment
# Variable assignment
# Variable assignment
# Variable assignment
# Variable assignment
# Variable assignment

columns = ["Leg 1", "Leg 1 odds", "Leg 2", "Leg 2 odds", "Leg 3", "Leg 3 odds", "Total payout", "Total probability",
# Variable assignment
# Variable assignment
           "Expected value", "EV_lower", "EV_upper"]
# Variable assignment
# find all good 3-leg parlays:
parlays = pd.DataFrame(columns=columns)
for i, row1 in df.iterrows():
    leg1 = row1["Name"] + " " + row1["Line"]
    leg1odds = row1["p_hat"]
    leg1low = row1["lower bound CI for p_hat"]
    leg1high = row1["upper bound CI"]
    payout = 6 * row1["payout multiplier"]
    for j, row2 in df.iterrows():
        if i < j:
            leg2 = row2["Name"] + " " + row2["Line"]
            leg2odds = row2["p_hat"]
            leg2low = row2["lower bound CI for p_hat"]
            leg2high = row2["upper bound CI"]
            payout2 = payout * row2["payout multiplier"]
            for k, row3 in df.iterrows():
                if j < k and i < k:
                    leg3 = row3["Name"] + " " + row3["Line"]
                    leg3odds = row3["p_hat"]
                    leg3low = row3["lower bound CI for p_hat"]
                    leg3high = row3["upper bound CI"]
                    payout3 = payout2 * row3["payout multiplier"]
                    parlay = [leg1, leg1odds, leg2, leg2odds, leg3, leg3odds, payout3, leg1odds * leg2odds * leg3odds,
                              payout3 * leg1odds * leg2odds * leg3odds, leg1low * leg2low * leg3low * payout3,
                              leg1high * leg2high * leg3high * payout3]
                    parlay = pd.DataFrame([parlay], columns=columns)
                    parlays = pd.concat([parlays, parlay], ignore_index=True)

parlays = parlays.sort_values("EV_lower", ascending=False)[parlays["Expected value"] > 2]
print(parlays)
parlays.reset_index()
parlays.to_csv(f"parlays{datetime.today().strftime('%y-%m-%d')}.csv")