## Scraping Premier League Data

# import libraries
import requests
import pandas as pd
from bs4 import BeautifulSoup
import time

# starting URL for latest season stats
prem_standings_url = "https://fbref.com/en/comps/9/Premier-League-Stats"

# loop over multiple seasons
years = list(range(2022, 2019, -1))  # scrape 2024 and 2023
prem_team_matches = []  # this will store all team data

# loop through each season
for year in years:
    data = requests.get(prem_standings_url)
    soup = BeautifulSoup(data.text, "html.parser")
    time.sleep(10)  # delay to avoid getting blocked

    # get all team links from the table
    prem_standings_table = soup.select('table.stats_table')[0]
    team_links = [l.get("href") for l in prem_standings_table.find_all('a') if '/squads/' in l.get("href")]

    # full team URLs
    team_urls = [f"https://fbref.com{l}" for l in team_links]

    # find link to previous season for the next loop
    previous_season = soup.select("a.prev")[0].get("href")
    prem_standings_url = f"https://fbref.com{previous_season}"  # update URL for next year

    # now scrape each team individually
    for team_url in team_urls:
        team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")

        # scrape match data
        team_data = requests.get(team_url)
        matches = pd.read_html(team_data.text, match="Scores & Fixtures")
        time.sleep(10)

        # get shooting stats link
        temp_soup = BeautifulSoup(team_data.text, "html.parser")
        temp_links = [l.get("href") for l in temp_soup.find_all('a') if 'all_comps/shooting/' in l.get("href")]
        if not temp_links:
            continue  # skip if no shooting data

        # scrape shooting data
        shooting_data = requests.get(f"https://fbref.com{temp_links[0]}")
        shooting = pd.read_html(shooting_data.text, match="Shooting")[0]
        shooting.columns = shooting.columns.droplevel()  # clean multi-index

        try:
            # merge fixtures with shooting stats
            team_df = matches[0].merge(
                shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]],
                on="Date"
            )
        except ValueError:
            continue  # skip if merging fails

        # only keep Premier League matches
        team_df = team_df[team_df["Comp"] == "Premier League"]
        team_df["Season"] = year
        team_df["Team"] = team_name
        prem_team_matches.append(team_df)
        time.sleep(10)  # avoid hammering the server - SUPER IMPORTANT - got blocked too many times to count

# combine all team data into one big DataFrame
match_df = pd.concat(prem_team_matches)

# convert all column names to lowercase for consistency
match_df.columns = [c.lower() for c in match_df.columns]

# export cleaned dataset to CSV for modeling later
match_df.to_csv("matches1.csv")
