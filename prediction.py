# Premier League Match Predictor: 2020–2022 Seasons

import pandas as pd 
matches = pd.read_csv("matches.csv", index_col=0)

## converting relevant columns into formats that can be used by machine learning models
matches["date"] = pd.to_datetime(matches["date"])
matches["h/a"] = matches["venue"].astype("category").cat.codes  # turning venue into 0 (away) or 1 (home)
matches["opp"] = matches["opponent"].astype("category").cat.codes  # turning opponent names into numeric codes
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")  # grabbing just the hour part from match time
matches["day"] = matches["date"].dt.dayofweek  # converting date to day of week (0=Monday, 6=Sunday)

matches["target"] = (matches["result"] == "W").astype("int")  # setting 'W' (win) as 1, everything else as 0

from sklearn.ensemble import RandomForestClassifier  # loading up random forest for classification

rf = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=1)
train = matches[matches["date"] < '2022-01-01'] 
test = matches[matches["date"] > '2022-01-01']
predictors = ["h/a", "opp", "hour", "day"]
rf.fit(train[predictors], train["target"])
RandomForestClassifier(min_samples_split=10, n_estimators=100, random_state=1)
preds = rf.predict(test[predictors])  # predicting match outcomes

from sklearn.metrics import accuracy_score
acc = accuracy_score(test["target"], preds)  # checking how accurate the model is
acc
combined = pd.DataFrame(dict(actual=test["target"], prediction=preds))
pd.crosstab(index=combined["actual"], columns=combined["prediction"])

from sklearn.metrics import precision_score
precision_score(test["target"], preds)

grouped_matches = matches.groupby("team") 
group = grouped_matches.get_group("Manchester United").sort_values("date")

def rolling_averages(group, cols, new_cols):  # function to calculate recent form (last 3 matches)
    group = group.sort_values("date")  # sort matches chronologically
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)  # drop rows where rolling averages couldn't be calculated
    return group 

cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"] 
new_cols = [f"{c}_rolling" for c in cols]  # new column names for the rolling average features

rolling_averages(group, cols, new_cols)  # running the function on a single team's data

matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling = matches_rolling.droplevel('team')  # cleaning up the multi-index after groupby

matches_rolling.index = range(matches_rolling.shape[0])  # resetting the index to be sequential
matches_rolling

def make_predictions(data, predictors):  # runs training + prediction in one go
    train = data[data["date"] < '2022-01-01'] 
    test = data[data["date"] > '2022-01-01']
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])  # get predicted results
    combined = pd.DataFrame(dict(actual=test["target"], prediction=preds), index=test.index)
    precision = precision_score(test["target"], preds)
    return combined, precision  # return prediction details and the precision score

combined, precision = make_predictions(matches_rolling, predictors + new_cols)

combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index=True, right_index=True)

class MissingDict(dict):  # custom dictionary that returns the key itself if not found
    __missing__ = lambda self, key: key

map_values = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Tottenham Hotspur": "Tottenham", 
    "West Ham United": "West Ham", 
    "Wolverhampton Wanderers": "Wolves"
}
mapping = MissingDict(**map_values)
mapping["West Ham United"]

combined["new_team"] = combined["team"].map(mapping)  # standardizing team names using the mapping


merged = combined.merge(combined, left_on=["date", "new_team"], right_on=["date", "opponent"]) ## finding both the home and away team predictions and merging them 

#Thank you DataQuest!
