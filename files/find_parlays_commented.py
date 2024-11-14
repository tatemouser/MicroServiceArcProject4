from datetime import datetime
import math

import pandas as pd

# Variable assignment
df = pd.read_csv(f"total_odds{datetime.today().strftime('%y-%m-%d')}.csv", index_col=0)
# Variable assignment
df = df.sort_values("P[p_hat <= p_needed]")[df["P[p_hat <= p_needed]"] < 0.2]
# Variable assignment
df = df.reset_index()

# Variable assignment
columns = ["Leg 1", "Leg 1 odds", "Leg 2", "Leg 2 odds", "Leg 3", "Leg 3 odds", "Total payout", "Total probability",
           "Expected value", "EV_lower", "EV_upper"]
# find all good 3-leg parlays:
# Variable assignment
parlays = pd.DataFrame(columns=columns)
# Loop starts here
for i, row1 in df.iterrows():
    # Variable assignment
    leg1 = row1["Name"] + " " + row1["Line"]
    # Variable assignment
    leg1odds = row1["p_hat"]
    # Variable assignment
    leg1low = row1["lower bound CI for p_hat"]
    # Variable assignment
    leg1high = row1["upper bound CI"]
    # Variable assignment
    payout = 6 * row1["payout multiplier"]
    # Loop starts here
    for j, row2 in df.iterrows():
        # Conditional statement
        if i < j:
            # Variable assignment
            leg2 = row2["Name"] + " " + row2["Line"]
            # Variable assignment
            leg2odds = row2["p_hat"]
            # Variable assignment
            leg2low = row2["lower bound CI for p_hat"]
            # Variable assignment
            leg2high = row2["upper bound CI"]
            # Variable assignment
            payout2 = payout * row2["payout multiplier"]
            # Loop starts here
            for k, row3 in df.iterrows():
                # Conditional statement
                if j < k and i < k:
                    # Variable assignment
                    leg3 = row3["Name"] + " " + row3["Line"]
                    # Variable assignment
                    leg3odds = row3["p_hat"]
                    # Variable assignment
                    leg3low = row3["lower bound CI for p_hat"]
                    # Variable assignment
                    leg3high = row3["upper bound CI"]
                    # Variable assignment
                    payout3 = payout2 * row3["payout multiplier"]
                    # Variable assignment
                    parlay = [leg1, leg1odds, leg2, leg2odds, leg3, leg3odds, payout3, leg1odds * leg2odds * leg3odds,
                              payout3 * leg1odds * leg2odds * leg3odds, leg1low * leg2low * leg3low * payout3,
                              leg1high * leg2high * leg3high * payout3]
                    # Variable assignment
                    parlay = pd.DataFrame([parlay], columns=columns)
                    # Variable assignment
                    parlays = pd.concat([parlays, parlay], ignore_index=True)

# Variable assignment
parlays = parlays.sort_values("EV_lower", ascending=False)[parlays["Expected value"] > 2]
print(parlays)
parlays.reset_index()
parlays.to_csv(f"parlays{datetime.today().strftime('%y-%m-%d')}.csv")