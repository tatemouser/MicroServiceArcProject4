import time

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import math
import scipy.stats as st
from datetime import datetime

pitcher_lines = ["Strikeouts", "Hits Allowed", "Walks Allowed", "Earned Runs Allowed", "Pitching Outs"]

hitter_lines = ["Total Bases", "Hits + Runs + RBIs", "Home Runs", "Singles", "Doubles", "Triples", "Stolen Bases",
                "Hits", "Runs", "RBIs"]

stat_dict = {"Strikeouts" : "SO", "Hits Allowed" : "H", "Walks Allowed" : "BB", "Earned Runs Allowed" : "ER",
             "Pitching Outs" : "Outs", "Total Bases" : "TB", "Hits + Runs + RBIs" : "H+R+RBI", "Home Runs" : "HR",
             "Singles" : "S", "Doubles" : "2B", "Triples" : "3B", "Stolen Bases" : "SB", "Hits" : "H", "Runs" : "R",
             "RBIs" : "RBI"}

columns = ["Name", "Line", "p_hat", "p_needed", "P[p_hat <= p_needed]", "lower bound CI for p_hat", "upper bound CI", "payout multiplier"]



def correct_table(data):
    if "FIP" in data.columns:  # is pitcher
        data = data[["IP", "H", "ER", "SO", "BB"]]
    else:  # is hitter
        data = data[["H", "2B", "3B", "HR", "RBI", "SB", "R"]]
    data = data.apply(pd.to_numeric, errors='coerce').dropna()
    return data[:-1]


def find_gamelog(row, from_disk=False):
    if not from_disk:
        name = row["Name"]
        first, last = name.split(" ")[0], "".join(name.split(" ")[1:])
        first = first[:2]
        if len(last) > 5:
            last = last[:5]

        if row["Line type"] in pitcher_lines:
            role = "p"
        elif row["Line type"] in hitter_lines:
            role = "b"
        else:
            return
        num = 1
        req = requests.get(
            f"https://www.baseball-reference.com/players/gl.fcgi?id={last}{first}0{str(num)}&t={role}&year=2024"
        ).content
        bs = BeautifulSoup(req, parser="lxml")
        print(req)
        if role == "p":
            table = bs.find("table", id="pitching_gamelogs")
        else:
            table = bs.find("table", id="batting_gamelogs")
        while table is None and num <= 3:
            time.sleep(5)
            num += 1
            req = requests.get(
                f"https://www.baseball-reference.com/players/gl.fcgi?id={last}{first}0{str(num)}&t={role}&year=2024").content
            bs = BeautifulSoup(req, parser="lxml", features="lxml")
            if role == "p":
                table = bs.find("table", id="pitching_gamelogs")
            else:
                table = bs.find("table", id="batting_gamelogs")
        if table is None:
            return
        tbl = pd.read_html(str(table))
        tbl = pd.DataFrame(tbl[0])
        print(tbl["Inngs"].str.contains("CG|GS", regex=True).count() / tbl["Inngs"].count())
        if (tbl["Inngs"].str.contains("CG|GS", regex=True).count() / tbl["Inngs"].count()) < 0.8: # if starting less than 80% of games in
            return
    else:
        tbl = pd.read_csv("gamelog_1.csv")
    return tbl


# Computing odds returns a tuple of: name, line, probability, needed probability for profit, p-value of hypothesis
# testing over the needed probability, and lower and upper 95% confidence interval bounds
def compute_odds(log, line, over, name, stat, implied_prob):
    # find column by stat, do manipulation
    col = stat_dict[stat]
    if col == "Outs":
        log["Outs"] = log["IP"] % 1 + (log["IP"].round() * 3)
    elif col == "TB":
        log["TB"] = log["H"] + log["2B"] + (2 * log["3B"]) + (3 * log["HR"])
    elif col == "H+R+RBI":
        log[col] = log["H"] + log["R"] + log["RBI"]
    elif col == "S":
        log[col] = log["H"] - log["2B"] - log["3B"] - log["HR"]


    # finding probability > 0.55
    n = log[[col]].count()
    if over:
        p_hat = log[log[col] > line].count()[[col]] / n
    else:
        p_hat = log[log[col] < line].count()[[col]] / n

    p_hat = p_hat.item()
    if p_hat == 1 or p_hat == 0:
        return
    needed_prob = math.pow(1/6, 1/3) / ((1 / implied_prob) - 1)
    print(needed_prob)
    if needed_prob == 0 or n.item() == 0:
        return

    z = (p_hat - needed_prob) / math.sqrt((needed_prob * (1 - needed_prob)) / n)
    p = st.norm.cdf(-z)

    lower = p_hat - (1.96 * math.sqrt(needed_prob * (1 - needed_prob) / n))
    upper = p_hat + (1.96 * math.sqrt(needed_prob * (1 - needed_prob) / n))

    total_line = f"{'Over' if over else 'Under'} {line} {stat}"

    return name, total_line, p_hat, needed_prob, p, lower, upper, (1 / implied_prob) - 1


def find_odds(df):
    final_odds = pd.DataFrame(columns=columns)
    for _, row in df.iterrows():
        gamelog = find_gamelog(row)
        if gamelog is None:
            continue
        gamelog = correct_table(gamelog)
        odds = compute_odds(gamelog, row["Line"], row["Over"], row["Name"], row["Line type"],
                            row["Implied probability"])
        if odds is None:
            continue
        odds = pd.DataFrame([odds], columns=columns)
        final_odds = pd.concat([final_odds, odds], ignore_index=True)
        print(odds)
        time.sleep(5)
    return final_odds


# debug to make them not mad at me
df = pd.read_csv(f"Odds{datetime.today().strftime('%y-%m-%d')}.csv", index_col=0)
total_odds = find_odds(df[:700])

total_odds.to_csv(f"total_odds{datetime.today().strftime('%y-%m-%d')}.csv")

