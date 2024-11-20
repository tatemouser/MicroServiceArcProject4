import time

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import math
import scipy.stats as st
from datetime import datetime

# Variable assignment
pitcher_lines = ["Strikeouts", "Hits Allowed", "Walks Allowed", "Earned Runs Allowed", "Pitching Outs"]

# Variable assignment
hitter_lines = ["Total Bases", "Hits + Runs + RBIs", "Home Runs", "Singles", "Doubles", "Triples", "Stolen Bases",
                "Hits", "Runs", "RBIs"]

# Variable assignment
stat_dict = {"Strikeouts" : "SO", "Hits Allowed" : "H", "Walks Allowed" : "BB", "Earned Runs Allowed" : "ER",
             "Pitching Outs" : "Outs", "Total Bases" : "TB", "Hits + Runs + RBIs" : "H+R+RBI", "Home Runs" : "HR",
             "Singles" : "S", "Doubles" : "2B", "Triples" : "3B", "Stolen Bases" : "SB", "Hits" : "H", "Runs" : "R",
             "RBIs" : "RBI"}

# Variable assignment
columns = ["Name", "Line", "p_hat", "p_needed", "P[p_hat <= p_needed]", "lower bound CI for p_hat", "upper bound CI", "payout multiplier"]



# Function 'correct_table' definition
def correct_table(data):
    # Conditional statement
    if "FIP" in data.columns:  # is pitcher
        # Variable assignment
        data = data[["IP", "H", "ER", "SO", "BB"]]
    else:  # is hitter
        # Variable assignment
        data = data[["H", "2B", "3B", "HR", "RBI", "SB", "R"]]
    # Variable assignment
    data = data.apply(pd.to_numeric, errors='coerce').dropna()
    return data[:-1]


# Function 'find_gamelog' definition
def find_gamelog(row, from_disk=False):
    # Conditional statement
    if not from_disk:
        # Variable assignment
        name = row["Name"]
        # Variable assignment
        first, last = name.split(" ")[0], "".join(name.split(" ")[1:])
        # Variable assignment
        first = first[:2]
        # Conditional statement
        if len(last) > 5:
            # Variable assignment
            last = last[:5]

        # Conditional statement
        if row["Line type"] in pitcher_lines:
            # Variable assignment
            role = "p"
        # Conditional statement
        elif row["Line type"] in hitter_lines:
            # Variable assignment
            role = "b"
        else:
            return
        # Variable assignment
        num = 1
        # Variable assignment
        req = requests.get(
            f"https://www.baseball-reference.com/players/gl.fcgi?id={last}{first}0{str(num)}&t={role}&year=2024"
        ).content
        # Variable assignment
        bs = BeautifulSoup(req, parser="lxml")
        print(req)
        # Conditional statement
        if role == "p":
            # Variable assignment
            table = bs.find("table", id="pitching_gamelogs")
        else:
            # Variable assignment
            table = bs.find("table", id="batting_gamelogs")
        while table is None and num <= 3:
            time.sleep(5)
            num += 1
            # Variable assignment
            req = requests.get(
                f"https://www.baseball-reference.com/players/gl.fcgi?id={last}{first}0{str(num)}&t={role}&year=2024").content
            # Variable assignment
            bs = BeautifulSoup(req, parser="lxml", features="lxml")
            # Conditional statement
            if role == "p":
                # Variable assignment
                table = bs.find("table", id="pitching_gamelogs")
            else:
                # Variable assignment
                table = bs.find("table", id="batting_gamelogs")
        # Conditional statement
        if table is None:
            return
        # Variable assignment
        tbl = pd.read_html(str(table))
        # Variable assignment
        tbl = pd.DataFrame(tbl[0])
        print(tbl["Inngs"].str.contains("CG|GS", regex=True).count() / tbl["Inngs"].count())
        # Conditional statement
        if (tbl["Inngs"].str.contains("CG|GS", regex=True).count() / tbl["Inngs"].count()) < 0.8: # if starting less than 80% of games in
            return
    else:
        # Variable assignment
        tbl = pd.read_csv("gamelog_1.csv")
    return tbl


# Computing odds returns a tuple of: name, line, probability, needed probability for profit, p-value of hypothesis
# testing over the needed probability, and lower and upper 95% confidence interval bounds
# Function 'compute_odds' definition
def compute_odds(log, line, over, name, stat, implied_prob):
    # find column by stat, do manipulation
    # Variable assignment
    col = stat_dict[stat]
    # Conditional statement
    if col == "Outs":
        # Variable assignment
        log["Outs"] = log["IP"] % 1 + (log["IP"].round() * 3)
    # Conditional statement
    elif col == "TB":
        # Variable assignment
        log["TB"] = log["H"] + log["2B"] + (2 * log["3B"]) + (3 * log["HR"])
    # Conditional statement
    elif col == "H+R+RBI":
        # Variable assignment
        log[col] = log["H"] + log["R"] + log["RBI"]
    # Conditional statement
    elif col == "S":
        # Variable assignment
        log[col] = log["H"] - log["2B"] - log["3B"] - log["HR"]


    # finding probability > 0.55
    # Variable assignment
    n = log[[col]].count()
    # Conditional statement
    if over:
        # Variable assignment
        p_hat = log[log[col] > line].count()[[col]] / n
    else:
        # Variable assignment
        p_hat = log[log[col] < line].count()[[col]] / n

    # Variable assignment
    p_hat = p_hat.item()
    # Conditional statement
    if p_hat == 1 or p_hat == 0:
        return
    # Variable assignment
    needed_prob = math.pow(1/6, 1/3) / ((1 / implied_prob) - 1)
    print(needed_prob)
    # Conditional statement
    if needed_prob == 0 or n.item() == 0:
        return

    # Variable assignment
    z = (p_hat - needed_prob) / math.sqrt((needed_prob * (1 - needed_prob)) / n)
    # Variable assignment
    p = st.norm.cdf(-z)

    # Variable assignment
    lower = p_hat - (1.96 * math.sqrt(needed_prob * (1 - needed_prob) / n))
    # Variable assignment
    upper = p_hat + (1.96 * math.sqrt(needed_prob * (1 - needed_prob) / n))

    # Variable assignment
    total_line = f"{'Over' if over else 'Under'} {line} {stat}"

    return name, total_line, p_hat, needed_prob, p, lower, upper, (1 / implied_prob) - 1


# Function 'find_odds' definition
def find_odds(df):
    # Variable assignment
    final_odds = pd.DataFrame(columns=columns)
    # Loop starts here
    for _, row in df.iterrows():
        # Variable assignment
        gamelog = find_gamelog(row)
        # Conditional statement
        if gamelog is None:
            continue
        # Variable assignment
        gamelog = correct_table(gamelog)
        # Variable assignment
        odds = compute_odds(gamelog, row["Line"], row["Over"], row["Name"], row["Line type"],
                            row["Implied probability"])
        # Conditional statement
        if odds is None:
            continue
        # Variable assignment
        odds = pd.DataFrame([odds], columns=columns)
        # Variable assignment
        final_odds = pd.concat([final_odds, odds], ignore_index=True)
        print(odds)
        time.sleep(5)
    return final_odds


# debug to make them not mad at me
# Variable assignment
df = pd.read_csv(f"Odds{datetime.today().strftime('%y-%m-%d')}.csv", index_col=0)
# Variable assignment
total_odds = find_odds(df[:700])

total_odds.to_csv(f"total_odds{datetime.today().strftime('%y-%m-%d')}.csv")

