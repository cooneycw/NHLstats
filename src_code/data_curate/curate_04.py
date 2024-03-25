import pandas as pd


def curate_rolling_player_stats(config, curr_date, first_days, days):
    player_df = pd.read_csv("storage/playerData.csv", index_col=None)
    player_df['gameDate'] = pd.to_datetime(player_df['gameDate'])
    player_df = player_df[player_df['positionGoalie'] == 0]
    player_df['games_played'] = 1

    basic_df = pd.read_csv("storage/game_data.csv", index_col=None)
    cols_to_keep = ['team', 'date', 'game_id']
    basic_df = basic_df.loc[:, cols_to_keep]
    basic_df.rename(columns={'game_id': 'gameId', 'date': 'gameDate'}, inplace=True)
    basic_df['gameDate'] = pd.to_datetime(basic_df['gameDate'])

    full_df = pd.read_csv("storage/game_data.csv", index_col=None)
    full_df.rename(columns={'game_id': 'gameId', 'date': 'gameDate'}, inplace=True)
    full_df['gameDate'] = pd.to_datetime(full_df['gameDate'])

    keep_cols = ['team', 'gameDate', 'opp', 'home', 'team_rest', 'opp_rest', 'team_last_game_date',
                 'opp_last_game_date',
                 ]
    keyword_columns = ["team_avg_wins_", "opp_avg_wins_",
                       "team_avg_goals_for_", "team_avg_goals_against_",
                       "opp_avg_goals_for_", "opp_avg_goals_against_",
                       "team_avg_shots_for_", "team_avg_shots_against_",
                       "opp_avg_shots_for_", "opp_avg_shots_against_",
                       "team_avg_pim_for_", "team_avg_pim_against_",
                       "opp_avg_pim_for_", "opp_avg_pim_against_",
                       "team_avg_blocks_for_", "team_avg_blocks_against_",
                       "opp_avg_blocks_for_", "opp_avg_blocks_against_",
                       "team_avg_hits_for_", "team_avg_hits_against_",
                       "opp_avg_hits_for_", "opp_avg_hits_against_",
                       "team_avg_face_for_", "team_avg_face_against_",
                       "opp_avg_face_for_", "opp_avg_face_against_",
                       ]

    for keyword in keyword_columns:
        keep_cols.extend(full_df.filter(like=keyword).columns.tolist())

    full_df = full_df[keep_cols]
    full_df.sort_values(by=['team', 'gameDate'], ascending=[True, False], inplace=True)

    unique_teams = basic_df['team'].unique()

    p_id = []
    p_lastName = []
    p_firstName = []
    p_team = []
    p_positionCentre = []
    p_positionRightWing = []
    p_positionLeftWing = []
    p_positionDefense = []
    p_gameDate = []
    p_anyGoals = []
    p_anyAssists = []
    p_anyPoints = []
    p_anyShots_01p = []
    p_anyShots_02p = []
    p_anyShots_03p = []
    p_anyShots_04p = []
    p_avg_toi = []
    p_avg_gamesPlayed = []
    p_avg_pim = []
    p_avg_hits = []
    p_avg_goals = []
    p_avg_anyGoals = []
    p_avg_assists = []
    p_avg_anyAssists = []
    p_avg_points = []
    p_avg_anyPoints = []
    p_avg_shots = []
    p_avg_anyShots_01p = []
    p_avg_anyShots_02p = []
    p_avg_anyShots_03p = []
    p_avg_anyShots_04p = []

    for team in unique_teams:
        print(f'Processing historic player data (days = {days}): {team}')
        team_df = basic_df[basic_df['team'] == team].copy()
        team_df.sort_values(by=['team', 'gameDate'], ascending=[True, False], inplace=True)
        for index, row in team_df.iterrows():
            # Extract the game date for the current row
            current_game_date = row['gameDate']
            prior_game_dates = team_df[team_df['gameDate'] < current_game_date]['gameDate']
            player_id_list = player_df[(player_df['team'] == team) & (player_df['gameDate'] == current_game_date)]['playerId'].unique()

            if len(prior_game_dates) >= days:
                # Extract the 20 most recent dates
                recent_dates = prior_game_dates.head(days)

                # Calculate minimum and maximum dates
                min_date = recent_dates.min()
                max_date = recent_dates.max()

                # Do whatever you need with min_date and max_date
                # print(f"For team {team}, minimum date among the 20 most recent dates is: {min_date}")
                # print(f"For team {team}, maximum date among the 20 most recent dates is: {max_date}")

                for player_id in player_id_list:
                    player_curr_game_df = player_df[(player_df['playerId'] == player_id) & (player_df['gameDate'] == current_game_date)]
                    player_prior_games_df = player_df[(player_df['playerId'] == player_id) & (player_df['gameDate'] >= min_date) & (player_df['gameDate'] <= max_date)]
                    if len(player_prior_games_df) < int(0.5 + 0.5 * days):
                        continue
                    p_id.append(player_id)
                    p_lastName.append(player_prior_games_df['playerLastName'].iloc[0])
                    p_firstName.append(player_prior_games_df['playerFirstName'].iloc[0])
                    p_positionCentre.append(player_curr_game_df['positionCentre'].iloc[0])
                    p_positionRightWing.append(player_curr_game_df['positionRightWing'].iloc[0])
                    p_positionLeftWing.append(player_curr_game_df['positionLeftWing'].iloc[0])
                    (p_positionDefense.append(player_curr_game_df['positionDefense'].iloc[0]))
                    p_team.append(team)
                    p_gameDate.append(current_game_date)
                    games_played = sum(player_prior_games_df['games_played'])
                    p_anyGoals.append(player_curr_game_df['anyGoals'].iloc[0])
                    p_anyAssists.append(player_curr_game_df['anyAssists'].iloc[0])
                    p_anyPoints.append(player_curr_game_df['anyPoints'].iloc[0])
                    p_anyShots_01p.append(player_curr_game_df['anyShots_01p'].iloc[0])
                    p_anyShots_02p.append(player_curr_game_df['anyShots_02p'].iloc[0])
                    p_anyShots_03p.append(player_curr_game_df['anyShots_03p'].iloc[0])
                    p_anyShots_04p.append(player_curr_game_df['anyShots_04p'].iloc[0])
                    p_avg_toi.append(sum(player_prior_games_df['toi']) / games_played)
                    p_avg_pim.append(sum(player_prior_games_df['pim']) / games_played)
                    p_avg_hits.append(sum(player_prior_games_df['hits']) / games_played)
                    p_avg_gamesPlayed.append(games_played)
                    p_avg_goals.append(sum(player_prior_games_df['goals'])/games_played)
                    p_avg_anyGoals.append(sum(player_prior_games_df['anyGoals']) / games_played)
                    p_avg_assists.append(sum(player_prior_games_df['assists']) / games_played)
                    p_avg_anyAssists.append(sum(player_prior_games_df['anyAssists']) / games_played)
                    p_avg_points.append(sum(player_prior_games_df['points']) / games_played)
                    p_avg_anyPoints.append(sum(player_prior_games_df['anyPoints']) / games_played)
                    p_avg_shots.append(sum(player_prior_games_df['shots']) / games_played)
                    p_avg_anyShots_01p.append(sum(player_prior_games_df['anyShots_01p']) / games_played)
                    p_avg_anyShots_02p.append(sum(player_prior_games_df['anyShots_02p']) / games_played)
                    p_avg_anyShots_03p.append(sum(player_prior_games_df['anyShots_03p']) / games_played)
                    p_avg_anyShots_04p.append(sum(player_prior_games_df['anyShots_04p']) / games_played)
            else:
                continue

    playerFeature_df = pd.DataFrame({
        f"player": p_id,
        f"playerLastName": p_lastName,
        f'playerFirstName': p_firstName,
        f"team": p_team,
        f"gameDate": p_gameDate,
        f"positionCentre": p_positionCentre,
        f"positionRightWing": p_positionRightWing,
        f"positionLeftWing": p_positionLeftWing,
        f"positionDefense": p_positionDefense,
        f"anyGoals": p_anyGoals,
        f"anyAssists": p_anyAssists,
        f"anyPoints": p_anyPoints,
        f"anyShots_01p": p_anyShots_01p,
        f"anyShots_02p": p_anyShots_02p,
        f"anyShots_03p": p_anyShots_03p,
        f"anyShots_04p": p_anyShots_04p,
        f"avg_gamesPlayed_{days}": p_avg_gamesPlayed,
        f"avg_toi_{days}": p_avg_toi,
        f"avg_pim_{days}": p_avg_pim,
        f"avg_hits_{days}": p_avg_hits,
        f"avg_goals_{days}": p_avg_goals,
        f"avg_anyGoals_{days}": p_avg_anyGoals,
        f"avg_assists_{days}": p_avg_assists,
        f"avg_anyAssists_{days}": p_avg_anyAssists,
        f"avg_points_{days}": p_avg_points,
        f"avg_anyPoints_{days}": p_avg_anyPoints,
        f"avg_gameShots_{days}": p_avg_shots,
        f"avg_anyShots_01p_{days}": p_avg_anyShots_01p,
        f"avg_anyShots_02p_{days}": p_avg_anyShots_02p,
        f"avg_anyShots_03p_{days}": p_avg_anyShots_03p,
        f"avg_anyShots_04p_{days}": p_avg_anyShots_04p,
    })

    if first_days is not True:
        playerFeature_df.drop(columns=['playerLastName', 'playerFirstName',
                                       'positionCentre', 'positionDefense',
                                       'positionRightWing', 'positionLeftWing',
                                       'anyGoals', 'anyAssists', 'anyPoints',
                                       'anyShots_01p',
                                       'anyShots_02p',
                                       'anyShots_03p',
                                       'anyShots_04p'], inplace=True)

    playerFeature_df.to_csv(f"storage/feature_player_data_{days}.csv", index=False)

    if first_days:
        result_df = pd.merge(playerFeature_df, full_df, how='left', on=['team', 'gameDate'])
        result_df.to_csv(f"storage/playerStatsData.csv", index=False)
    else:
        result_df = pd.read_csv("storage/playerStatsData.csv", index_col=None)
        result_df['gameDate'] = pd.to_datetime(result_df['gameDate'])
        print(f'len of playerStatsData: {len(result_df)}')
        result_df = pd.merge(result_df, playerFeature_df, how='inner', on=['player', 'team', 'gameDate'])
        print(f'len of playerStatsData post merge: {len(result_df)}')
        result_df.to_csv(f"storage/playerStatsData.csv", index=False)


def curate_proj_player_data(config, curr_date, first_days, days):
    basic_df = pd.read_csv("storage/future_player.csv", index_col=None)
    basic_df['nextGameDate'] = pd.to_datetime(basic_df['gameDate'])
    basic_df['gameDate'] = pd.to_datetime(basic_df['team_last_game_date'])
    basic_df.sort_values(by=['team', 'gameDate'], inplace=True)

    complete_df = pd.read_csv("storage/game_data.csv", index_col=None)

    keep_cols = ['team', 'date', 'opp', 'home', 'team_rest', 'opp_rest', 'team_last_game_date',
                 'opp_last_game_date',
                 ]
    keyword_columns = ["team_avg_wins_", "opp_avg_wins_",
                       "team_avg_goals_for_", "team_avg_goals_against_",
                       "opp_avg_goals_for_", "opp_avg_goals_against_",
                       "team_avg_shots_for_", "team_avg_shots_against_",
                       "opp_avg_shots_for_", "opp_avg_shots_against_",
                       "team_avg_pim_for_", "team_avg_pim_against_",
                       "opp_avg_pim_for_", "opp_avg_pim_against_",
                       "team_avg_blocks_for_", "team_avg_blocks_against_",
                       "opp_avg_blocks_for_", "opp_avg_blocks_against_",
                       "team_avg_hits_for_", "team_avg_hits_against_",
                       "opp_avg_hits_for_", "opp_avg_hits_against_",
                       "team_avg_face_for_", "team_avg_face_against_",
                       "opp_avg_face_for_", "opp_avg_face_against_",
                       ]

    for keyword in keyword_columns:
        keep_cols.extend(complete_df.filter(like=keyword).columns.tolist())

    complete_df = complete_df.loc[:, keep_cols]
    complete_df.rename(columns={'game_id': 'gameId', 'date': 'gameDate'}, inplace=True)
    complete_df['gameDate'] = pd.to_datetime(complete_df['gameDate'])

    player_df = pd.read_csv("storage/playerData.csv", index_col=None)
    player_df['gameDate'] = pd.to_datetime(player_df['gameDate'])
    player_df = player_df[player_df['positionGoalie'] == 0]
    player_df['games_played'] = 1

    unique_teams = basic_df['team'].unique()

    p_id = []
    p_lastName = []
    p_firstName = []
    p_team = []
    p_positionCentre = []
    p_positionRightWing = []
    p_positionLeftWing = []
    p_positionDefense = []
    p_gameDate = []
    p_avg_toi = []
    p_avg_pim = []
    p_avg_hits = []
    p_avg_gamesPlayed = []
    p_avg_goals = []
    p_avg_anyGoals = []
    p_avg_assists = []
    p_avg_anyAssists = []
    p_avg_points = []
    p_avg_anyPoints = []
    p_avg_shots = []
    p_avg_anyShots_01p = []
    p_avg_anyShots_02p = []
    p_avg_anyShots_03p = []
    p_avg_anyShots_04p = []

    for team in unique_teams:
        print(f'Processing future player data (days = {days}): {team}')
        team_df = basic_df[basic_df['team'] == team].copy()
        team_df.sort_values(by=['team', 'gameDate'], ascending=[True, False], inplace=True)
        complete_df.sort_values(by=['team', 'gameDate'], ascending=[True, False], inplace=True)
        for team in unique_teams:
            # Extract the game date for the current row
            current_game_date = pd.to_datetime(curr_date)
            prior_game_dates = complete_df[(complete_df['gameDate'] < current_game_date) & (complete_df['team'] == team)]['gameDate']
            player_id_list = team_df[(team_df['team'] == team)]['player'].unique()

            if len(prior_game_dates) >= days:
                # Extract the 20 most recent dates
                recent_dates = prior_game_dates.head(days)

                # Calculate minimum and maximum dates
                min_date = recent_dates.min()
                max_date = recent_dates.max()

                # Do whatever you need with min_date and max_date
                # print(f"For team {team}, minimum date among the 20 most recent dates is: {min_date}")
                # print(f"For team {team}, maximum date among the 20 most recent dates is: {max_date}")

                for player_id in player_id_list:
                    player_prior_games_df = player_df[(player_df['playerId'] == player_id) & (player_df['gameDate'] >= min_date) & (player_df['gameDate'] <= max_date)]
                    if len(player_prior_games_df) < int(0.5 + 0.5 * days) or player_prior_games_df['positionGoalie'].iloc[0] == 1:
                        continue
                    p_id.append(player_id)
                    p_lastName.append(player_prior_games_df['playerLastName'].iloc[0])
                    p_firstName.append(player_prior_games_df['playerFirstName'].iloc[0])
                    p_positionCentre.append(player_prior_games_df['positionCentre'].iloc[0])
                    p_positionRightWing.append(player_prior_games_df['positionRightWing'].iloc[0])
                    p_positionLeftWing.append(player_prior_games_df['positionLeftWing'].iloc[0])
                    p_positionDefense.append(player_prior_games_df['positionDefense'].iloc[0])
                    p_team.append(team)
                    p_gameDate.append(current_game_date)
                    games_played = sum(player_prior_games_df['games_played'])
                    p_avg_toi.append(sum(player_prior_games_df['toi']) / games_played)
                    p_avg_pim.append(sum(player_prior_games_df['pim']) / games_played)
                    p_avg_hits.append(sum(player_prior_games_df['hits']) / games_played)
                    p_avg_gamesPlayed.append(games_played)
                    p_avg_goals.append(sum(player_prior_games_df['goals'])/games_played)
                    p_avg_anyGoals.append(sum(player_prior_games_df['anyGoals']) / games_played)
                    p_avg_assists.append(sum(player_prior_games_df['assists']) / games_played)
                    p_avg_anyAssists.append(sum(player_prior_games_df['anyAssists']) / games_played)
                    p_avg_points.append(sum(player_prior_games_df['points']) / games_played)
                    p_avg_anyPoints.append(sum(player_prior_games_df['anyPoints']) / games_played)
                    p_avg_shots.append(sum(player_prior_games_df['shots']) / games_played)
                    p_avg_anyShots_01p.append(sum(player_prior_games_df['anyShots_01p']) / games_played)
                    p_avg_anyShots_02p.append(sum(player_prior_games_df['anyShots_02p']) / games_played)
                    p_avg_anyShots_03p.append(sum(player_prior_games_df['anyShots_03p']) / games_played)
                    p_avg_anyShots_04p.append(sum(player_prior_games_df['anyShots_04p']) / games_played)
            else:
                continue

    playerFeature_df = pd.DataFrame({
        f"player": p_id,
        f"playerLastName": p_lastName,
        f'playerFirstName': p_firstName,
        f"team": p_team,
        f"gameDate": p_gameDate,
        f"positionCentre": p_positionCentre,
        f"positionRightWing": p_positionRightWing,
        f"positionLeftWing": p_positionLeftWing,
        f"positionDefense": p_positionDefense,
        f"avg_gamesPlayed_{days}": p_avg_gamesPlayed,
        f"avg_toi_{days}": p_avg_toi,
        f"avg_pim_{days}": p_avg_pim,
        f"avg_hits_{days}": p_avg_hits,
        f"avg_goals_{days}": p_avg_goals,
        f"avg_anyGoals_{days}": p_avg_anyGoals,
        f"avg_assists_{days}": p_avg_assists,
        f"avg_anyAssists_{days}": p_avg_anyAssists,
        f"avg_points_{days}": p_avg_points,
        f"avg_anyPoints_{days}": p_avg_anyPoints,
        f"avg_shots_{days}": p_avg_shots,
        f"avg_anyShots_01p_{days}": p_avg_anyShots_01p,
        f"avg_anyShots_02p_{days}": p_avg_anyShots_02p,
        f"avg_anyShots_03p_{days}": p_avg_anyShots_03p,
        f"avg_anyShots_04p_{days}": p_avg_anyShots_04p,
    })

    if first_days is not True:
        playerFeature_df.drop(columns=['playerLastName', 'playerFirstName',
                                       'positionCentre', 'positionDefense',
                                       'positionRightWing', 'positionLeftWing'], inplace=True)

    if first_days:
        full_df = pd.read_csv("storage/future_games.csv", index_col=None)
        full_df.rename(columns={'game_id': 'gameId', 'date': 'gameDate'}, inplace=True)
        full_df['gameDate'] = pd.to_datetime(full_df['gameDate'])

        keep_cols = ['team', 'gameDate', 'opp', 'home', 'team_rest', 'opp_rest', 'team_last_game_date',
                     'opp_last_game_date',
                     ]
        keyword_columns = ["team_avg_wins_", "opp_avg_wins_",
                           "team_avg_goals_for_", "team_avg_goals_against_",
                           "opp_avg_goals_for_", "opp_avg_goals_against_",
                           "team_avg_shots_for_", "team_avg_shots_against_",
                           "opp_avg_shots_for_", "opp_avg_shots_against_",
                           "team_avg_pim_for_", "team_avg_pim_against_",
                           "opp_avg_pim_for_", "opp_avg_pim_against_",
                           "team_avg_blocks_for_", "team_avg_blocks_against_",
                           "opp_avg_blocks_for_", "opp_avg_blocks_against_",
                           "team_avg_hits_for_", "team_avg_hits_against_",
                           "opp_avg_hits_for_", "opp_avg_hits_against_",
                           "team_avg_face_for_", "team_avg_face_against_",
                           "opp_avg_face_for_", "opp_avg_face_against_",
                           ]

        for keyword in keyword_columns:
            keep_cols.extend(full_df.filter(like=keyword).columns.tolist())

        full_df = full_df[keep_cols]
        full_df.sort_values(by=['team', 'gameDate'], ascending=[True, False], inplace=True)

        result_df = pd.merge(playerFeature_df, full_df, how='left', on=['team', 'gameDate'])
        result_df.to_csv(f"storage/playerFutureStatsData.csv", index=False)
    else:
        result_df = pd.read_csv("storage/playerFutureStatsData.csv", index_col=None)
        result_df['gameDate'] = pd.to_datetime(result_df['gameDate'])
        print(f'len of playerFutureStatsData: {len(result_df)}')
        result_df = pd.merge(result_df, playerFeature_df, how='inner', on=['player', 'team', 'gameDate'])
        print(f'len of playerFutureStatsData post merge: {len(result_df)}')
        result_df.to_csv(f"storage/playerFutureStatsData.csv", index=False)
