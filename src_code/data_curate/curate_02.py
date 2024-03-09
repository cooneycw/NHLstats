import pandas as pd


def curate_rolling_stats(config, curr_date, days):
    basic_df = pd.read_csv("storage/game_data.csv", index_col=None)
    basic_df.sort_values(by=['team', 'date'], inplace=True)
    # Assuming df is already defined and sorted by date

    # Calculate rolling metrics for each team
    last_team = None
    team = []
    date = []
    team_wins = []
    team_goals_for = []
    team_goals_against = []
    team_shots_for = []
    team_shots_against = []
    team_pim_for = []
    team_pim_against = []
    team_blocks_for = []
    team_blocks_against = []
    team_hits_for = []
    team_hits_against = []
    team_face_for = []
    team_face_against = []
    avg_wins = []
    avg_goals_for = []
    avg_goals_against = []
    avg_shots_for = []
    avg_shots_against = []
    avg_pim_for = []
    avg_pim_against = []
    avg_blocks_for = []
    avg_blocks_against = []
    avg_hits_for = []
    avg_hits_against = []
    avg_face_for = []
    avg_face_against = []
    for index, row in basic_df.iterrows():
        if row['team'] != last_team:
            team_wins = []
            team_goals_for = []
            team_goals_against = []
            team_shots_for = []
            team_shots_against = []
            team_pim_for = []
            team_pim_against = []
            team_blocks_for = []
            team_blocks_against = []
            team_hits_for = []
            team_hits_against = []
            team_face_for = []
            team_face_against = []
        team_wins.append(row['non_shootout_win'])
        team_goals_for.append(row['team_score'])
        team_goals_against.append(row['opp_score'])
        team_shots_for.append(row['team_sog'])
        team_shots_against.append(row['opp_sog'])
        team_pim_for.append(row['team_pim'])
        team_pim_against.append(row['opp_pim'])
        team_blocks_for.append(row['team_blocks'])
        team_blocks_against.append(row['opp_blocks'])
        team_hits_for.append(row['team_hits'])
        team_hits_against.append(row['opp_hits'])
        team_face_for.append(row['team_face'])
        team_face_against.append(row['opp_face'])

        if len(team_goals_for) >= days:
            team.append(row['team'])
            date.append(row['date'])
            avg_wins.append(sum(team_wins) / len(team_wins))
            avg_goals_for.append(sum(team_goals_for) / len(team_goals_for))
            avg_goals_against.append(sum(team_goals_against) / len(team_goals_against))
            avg_shots_for.append(sum(team_shots_for) / len(team_shots_for))
            avg_shots_against.append(sum(team_shots_against) / len(team_shots_against))
            avg_pim_for.append(sum(team_pim_for) / len(team_pim_for))
            avg_pim_against.append(sum(team_pim_against) / len(team_pim_against))
            avg_blocks_for.append(sum(team_blocks_for) / len(team_blocks_for))
            avg_blocks_against.append(sum(team_blocks_against) / len(team_blocks_against))
            avg_hits_for.append(sum(team_hits_for) / len(team_hits_for))
            avg_hits_against.append(sum(team_hits_against) / len(team_hits_against))
            avg_face_for.append(sum(team_face_for) / len(team_face_for))
            avg_face_against.append(sum(team_face_against) / len(team_face_against))

            team_wins.pop(0)
            team_goals_for.pop(0)
            team_goals_against.pop(0)
            team_shots_for.pop(0)
            team_shots_against.pop(0)
            team_pim_for.pop(0)
            team_pim_against.pop(0)
            team_blocks_for.pop(0)
            team_blocks_against.pop(0)
            team_hits_for.pop(0)
            team_hits_against.pop(0)
            team_face_for.pop(0)
            team_face_against.pop(0)

        last_team = row['team']

    feature_team_df = pd.DataFrame({
        "team": team,
        "team_last_game_date": date,
        f"team_avg_wins": avg_wins,
        f"team_avg_goals_for_{days}": avg_goals_for,
        f"team_avg_goals_against_{days}": avg_goals_against,
        f"team_avg_shots_for_{days}": avg_shots_for,
        f"team_avg_shots_against_{days}": avg_shots_against,
        f"team_avg_pim_for_{days}": avg_pim_for,
        f"team_avg_pim_against_{days}": avg_pim_against,
        f"team_avg_blocks_for_{days}": avg_blocks_for,
        f"team_avg_blocks_against_{days}": avg_blocks_against,
        f"team_avg_hits_for_{days}": avg_hits_for,
        f"team_avg_hits_against_{days}": avg_hits_against,
        f"team_avg_face_for_{days}": avg_face_for,
        f"team_avg_face_against_{days}": avg_face_against,
    })

    feature_opp_df = pd.DataFrame({
        "opp": team,
        "opp_last_game_date": date,
        f"opp_avg_wins": avg_wins,
        f"opp_avg_goals_for_{days}": avg_goals_for,
        f"opp_avg_goals_against_{days}": avg_goals_against,
        f"opp_avg_shots_for_{days}": avg_shots_for,
        f"opp_avg_shots_against_{days}": avg_shots_against,
        f"opp_avg_pim_for_{days}": avg_pim_for,
        f"opp_avg_pim_against_{days}": avg_pim_against,
        f"opp_avg_blocks_for_{days}": avg_blocks_for,
        f"opp_avg_blocks_against_{days}": avg_blocks_against,
        f"opp_avg_hits_for_{days}": avg_hits_for,
        f"opp_avg_hits_against_{days}": avg_hits_against,
        f"opp_avg_face_for_{days}": avg_face_for,
        f"opp_avg_face_against_{days}": avg_face_against,
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


