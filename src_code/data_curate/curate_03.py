from datetime import datetime
import pandas as pd
pd.set_option('display.max_columns', None)


def curate_player_stats(config, curr_date):
    boxscore_data = config.get_boxscores()
    player_data = config.get_player_list()
    players = dict()
    for player in player_data:
        players[player[1]] = dict()
        players[player[1]]['team'] = player[0]
        players[player[1]]['position'] = player[2]
        players[player[1]]['lastName'] = player[3]['default']
        players[player[1]]['firstName'] = player[4]['default']

    boxscore_set = set()
    player_id = []
    playerLastName = []
    playerFirstName = []
    team = []
    opp_team = []
    season = []
    gameId = []
    gameDate = []
    gamePlayed = []
    home = []
    positionCentre = []
    positionRightWing = []
    positionLeftWing = []
    positionDefense = []
    positionGoalie = []
    goals = []
    anyGoals = []
    assists = []
    anyAssists = []
    points = []
    anyPoints = []
    plusMinus = []
    pim = []
    hits = []
    powerPlayGoals = []
    shots = []
    anyShots_01p = []
    anyShots_02p = []
    anyShots_03p = []
    anyShots_04p = []
    faceoffWinningPctg = []
    toi = []
    gameEvenStrengthShotsAgainst = []
    gamePowerPlayShotsAgainst = []
    gameShorthandedShotsAgainst = []
    gameSaveShotsAgainst = []
    gameEvenStrengthGoalsAgainst = []
    gamePowerPlayGoalsAgainst = []
    gameShorthandedGoalsAgainst = []
    gameGoalsAgainst = []

    for boxscore in boxscore_data:
        t_season = boxscore[0]
        t_game_id = boxscore[2]
        t_date = boxscore[3]
        data = boxscore[4]
        game_date = datetime.strptime(t_date, '%Y-%m-%d').date()
        if game_date >= curr_date:
            continue
        for teamType in data['playerByGameStats'].keys():
            sub_data = data['playerByGameStats'][teamType]
            for playerType in sub_data.keys():
                sub_sub_data = sub_data[playerType]
                for element in sub_sub_data:
                    gameKey = (element['playerId'], boxscore[2])
                    if gameKey in boxscore_set:
                        continue
                    else:
                        boxscore_set.add(gameKey)
                    player_id.append(element['playerId'])
                    if players.get(element['playerId'], None) is None:
                        playerLastName.append('Not Active')
                        playerFirstName.append('Not Active')
                    else:
                        playerLastName.append(players[element['playerId']]['lastName'])
                        playerFirstName.append(players[element['playerId']]['firstName'])
                    team.append(data[teamType]['abbrev'])
                    gamePlayed.append(1)
                    if teamType == 'awayTeam':
                        opp_team.append(data['homeTeam']['abbrev'])
                    else:
                        opp_team.append(data['awayTeam']['abbrev'])
                    season.append(t_season)
                    gameId.append(t_game_id)
                    gameDate.append(t_date)
                    if teamType == 'homeTeam':
                        home.append(1)
                    else:
                        home.append(0)
                    if element['position'] == 'C':
                        positionCentre.append(1)
                        positionRightWing.append(0)
                        positionLeftWing.append(0)
                        positionDefense.append(0)
                        positionGoalie.append(0)
                    elif element['position'] == 'R':
                        positionCentre.append(0)
                        positionRightWing.append(1)
                        positionLeftWing.append(0)
                        positionDefense.append(0)
                        positionGoalie.append(0)
                    elif element['position'] == 'L':
                        positionCentre.append(0)
                        positionRightWing.append(0)
                        positionLeftWing.append(1)
                        positionDefense.append(0)
                        positionGoalie.append(0)
                    elif element['position'] == 'D':
                        positionCentre.append(0)
                        positionRightWing.append(0)
                        positionLeftWing.append(0)
                        positionDefense.append(1)
                        positionGoalie.append(0)
                    elif element['position'] == 'G':
                        positionCentre.append(0)
                        positionRightWing.append(0)
                        positionLeftWing.append(0)
                        positionDefense.append(0)
                        positionGoalie.append(1)

                    pim.append(element['pim'])
                    minutes, seconds = map(int, element['toi'].split(':'))
                    toi.append(round(minutes + (seconds/60), 2))
                    if element['position'] == 'G':
                        goals.append(0)
                        anyGoals.append(0)
                        assists.append(0)
                        anyAssists.append(0)
                        points.append(0)
                        anyPoints.append(0)
                        plusMinus.append(0)
                        hits.append(0)
                        powerPlayGoals.append(0)
                        shots.append(0)
                        anyShots_01p.append(0)
                        anyShots_02p.append(0)
                        anyShots_03p.append(0)
                        anyShots_04p.append(0)
                        faceoffWinningPctg.append(0)
                        gameEvenStrengthShotsAgainst.append(element['evenStrengthShotsAgainst'])
                        gamePowerPlayShotsAgainst.append(element['powerPlayShotsAgainst'])
                        gameShorthandedShotsAgainst.append(element['shorthandedShotsAgainst'])
                        gameSaveShotsAgainst.append(element['saveShotsAgainst'])
                        gameEvenStrengthGoalsAgainst.append(element['evenStrengthGoalsAgainst'])
                        gamePowerPlayGoalsAgainst.append(element['powerPlayGoalsAgainst'])
                        gameShorthandedGoalsAgainst.append(element['shorthandedGoalsAgainst'])
                        gameGoalsAgainst.append(element['goalsAgainst'])
                    else:
                        goals.append(element['goals'])
                        assists.append(element['assists'])
                        points.append(element['points'])
                        plusMinus.append(element['plusMinus'])
                        hits.append(element['hits'])
                        powerPlayGoals.append(element['powerPlayGoals'])
                        shots.append(element['shots'])
                        faceoffWinningPctg.append(element['faceoffWinningPctg'])
                        gameEvenStrengthShotsAgainst.append(0)
                        gamePowerPlayShotsAgainst.append(0)
                        gameShorthandedShotsAgainst.append(0)
                        gameSaveShotsAgainst.append(0)
                        gameEvenStrengthGoalsAgainst.append(0)
                        gamePowerPlayGoalsAgainst.append(0)
                        gameShorthandedGoalsAgainst.append(0)
                        gameGoalsAgainst.append(0)
                        if element['assists'] > 0:
                            anyAssists.append(1)
                        else:
                            anyAssists.append(0)
                        if element['goals'] > 0:
                            anyGoals.append(1)
                        else:
                            anyGoals.append(0)

                        if element['points'] > 0:
                            anyPoints.append(1)
                        else:
                            anyPoints.append(0)

                        t_01p = 0
                        t_02p = 0
                        t_03p = 0
                        t_04p = 0

                        if element['shots'] > 1:
                            t_01p = 1

                        if element['shots'] > 2:
                            t_02p = 1

                        if element['shots'] > 3:
                            t_03p = 1

                        if element['shots'] > 4:
                            t_04p = 1

                        anyShots_01p.append(t_01p)
                        anyShots_02p.append(t_02p)
                        anyShots_03p.append(t_03p)
                        anyShots_04p.append(t_04p)


    df = pd.DataFrame({
        "playerId": player_id,
        "playerLastName": playerLastName,
        "playerFirstName": playerFirstName,
        "season": season,
        "gameId": gameId,
        "gameDate": gameDate,
        "team": team,
        "opp": opp_team,
        "home": home,
        "gamePlayed": gamePlayed,
        "positionCentre": positionCentre,
        "positionRightWing": positionRightWing,
        "positionLeftWing": positionLeftWing,
        "positionDefense": positionDefense,
        "positionGoalie": positionGoalie,
        "goals": goals,
        "anyGoals": anyGoals,
        "assists": assists,
        "anyAssists": anyAssists,
        "points": points,
        "anyPoints": anyPoints,
        "plusMinus": plusMinus,
        "pim": pim,
        "hits": hits,
        "powerPlayGoals": powerPlayGoals,
        "shots": shots,
        "anyShots_01p": anyShots_01p,
        "anyShots_02p": anyShots_02p,
        "anyShots_03p": anyShots_03p,
        "anyShots_04p": anyShots_04p,
        "faceoffWinningPctg": faceoffWinningPctg,
        "toi": toi,
        "gameEvenStrengthShotsAgainst": gameEvenStrengthShotsAgainst,
        "gamePowerPlayShotsAgainst": gamePowerPlayShotsAgainst,
        "gameShorthandedShotsAgainst": gameShorthandedShotsAgainst,
        "gameSaveShotsAgainst": gameSaveShotsAgainst,
        "gameEvenStrengthGoalsAgainst": gameEvenStrengthGoalsAgainst,
        "gamePowerPlayGoalsAgainst": gamePowerPlayGoalsAgainst,
        "gameShorthandedGoalsAgainst": gameShorthandedGoalsAgainst,
        "gameGoalsAgainst": gameGoalsAgainst,
    })

    df['gameDate'] = pd.to_datetime(df['gameDate'])
    df['playerLastGameDate'] = df['gameDate']
    df.sort_values(by=['playerId', 'gameDate'], inplace=True)
    lastPlayer = None
    lastGameDate = None

    for index, row in df.iterrows():
        if lastPlayer == row['playerId']:
            df.loc[index, 'playerLastGameDate'] = lastGameDate
        else:
            df.loc[index, 'playerLastGameDate'] = None
        lastGameDate = row['gameDate']
        lastPlayer = row['playerId']

    df['playerRest'] = (df['gameDate'] - df['playerLastGameDate']).dt.days - 1

    df.to_csv("storage/playerData.csv", index=False)


def curate_future_player_stats(config, curr_date):
    roster_data = config.get_rosters()
    curr_date = pd.to_datetime(curr_date)

    t_team = []
    t_player_id = []
    t_player_lname = []
    t_player_fname = []
    t_positionCentre = []
    t_positionRightWing = []
    t_positionLeftWing = []
    t_positionDefense = []
    t_positionGoalie = []

    for team in roster_data:
        for position in team[1].keys():
            for player in team[1][position]:
                t_team.append(team[0])
                t_player_id.append(player['id'])
                t_player_lname.append(player['lastName']['default'])
                t_player_fname.append(player['firstName']['default'])
                if player['positionCode'] == 'C':
                    t_positionCentre.append(1)
                    t_positionRightWing.append(0)
                    t_positionLeftWing.append(0)
                    t_positionDefense.append(0)
                    t_positionGoalie.append(0)
                elif player['positionCode'] == 'R':
                    t_positionCentre.append(0)
                    t_positionRightWing.append(1)
                    t_positionLeftWing.append(0)
                    t_positionDefense.append(0)
                    t_positionGoalie.append(0)
                elif player['positionCode'] == 'L':
                    t_positionCentre.append(0)
                    t_positionRightWing.append(0)
                    t_positionLeftWing.append(1)
                    t_positionDefense.append(0)
                    t_positionGoalie.append(0)
                elif player['positionCode'] == 'D':
                    t_positionCentre.append(0)
                    t_positionRightWing.append(0)
                    t_positionLeftWing.append(0)
                    t_positionDefense.append(1)
                    t_positionGoalie.append(0)
                elif player['positionCode'] == 'G':
                    t_positionCentre.append(0)
                    t_positionRightWing.append(0)
                    t_positionLeftWing.append(0)
                    t_positionDefense.append(0)
                    t_positionGoalie.append(1)

    future_player_df = pd.DataFrame({
        "player": t_player_id,
        "team": t_team,
        "positionGoalie": t_positionGoalie,
    })

    future_player_df = future_player_df[future_player_df['positionGoalie'] == 0]
    future_player_df.drop(columns=['positionGoalie'], inplace=True)
    basic_df = pd.read_csv("storage/future_games.csv", index_col=None)
    basic_df.rename(columns={'game_id': 'gameId', 'date': 'gameDate'}, inplace=True)
    basic_df['gameDate'] = pd.to_datetime(basic_df['gameDate'])
    basic_df.sort_values(by=['team', 'gameDate'], inplace=True)
    basic_df = basic_df[basic_df['gameDate'] == curr_date]

    keep_cols = ['team', 'gameDate', 'opp', 'home', 'team_rest', 'opp_rest', 'team_last_game_date',
                 'opp_last_game_date']

    result_df = pd.merge(basic_df[keep_cols], future_player_df, how='left', on=['team'])
    result_df.to_csv("storage/future_player.csv", index=False)
