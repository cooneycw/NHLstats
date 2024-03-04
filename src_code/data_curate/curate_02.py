import pandas as pd


def curate_rolling_stats(config, curr_date, days):
    basic_df = pd.read_csv("storage/game_data.csv", index_col=None)
    basic_df.sort_values(by=['team', 'date'], inplace=True)
    # Assuming df is already defined and sorted by date

    # Calculate rolling metrics for each team
    last_team = None
    team = []
    date = []
    team_goals_for = []
    team_goals_against = []
    avg_goals_for = []
    avg_goals_against = []
    for index, row in basic_df.iterrows():
        if row['team'] != last_team:
            team_goals_for = []
            team_goals_against = []
        team_goals_for.append(row['team_score'])
        team_goals_against.append(row['opp_score'])

        if len(team_goals_for) >= days:
            team.append(row['team'])
            date.append(row['date'])
            avg_goals_for.append(sum(team_goals_for) / len(team_goals_for))
            avg_goals_against.append(sum(team_goals_against) / len(team_goals_against))
            team_goals_for.pop(0)
            team_goals_against.pop(0)

        last_team = row['team']

    feature_team_df = pd.DataFrame({
        "team": team,
        "team_last_game_date": date,
        f"team_avg_goals_for_{days}": avg_goals_for,
        f"team_avg_goals_against_{days}": avg_goals_against,
    })

    feature_opp_df = pd.DataFrame({
        "opp": team,
        "opp_last_game_date": date,
        f"opp_avg_goals_for_{days}": avg_goals_for,
        f"opp_avg_goals_against_{days}": avg_goals_against,
    })

    feature_team_df.to_csv(f"storage/feature_team_data_{days}.csv", index=False)
    feature_opp_df.to_csv(f"storage/feature_opp_data_{days}.csv", index=False)

    result_df = pd.merge(basic_df, feature_team_df, how='inner', on=['team', 'team_last_game_date'])
    result_df = pd.merge(result_df, feature_opp_df, how='inner', on=['opp', 'opp_last_game_date'])

    result_df.to_csv(f"storage/game_data.csv", index=False)


def curate_proj_data(config, curr_date, days):
    basic_df = pd.read_csv("storage/future_games.csv", index_col=None)
    basic_df.sort_values(by=['team', 'date'], inplace=True)
    feature_team_df = pd.read_csv(f"storage/feature_team_data_{days}.csv", index_col=None)
    feature_opp_df = pd.read_csv(f"storage/feature_opp_data_{days}.csv", index_col=None)

    result_df = pd.merge(basic_df, feature_team_df, how='inner', on=['team', 'team_last_game_date'])
    result_df = pd.merge(result_df, feature_opp_df, how='inner', on=['opp', 'opp_last_game_date'])

    result_df.to_csv(f"storage/future_games.csv", index=False)


