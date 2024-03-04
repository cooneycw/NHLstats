from datetime import datetime
import pandas as pd
pd.set_option('display.max_columns', None)


def curate_basic_stats(config, curr_date):
    data = config.get_boxscores()
    season = []
    game_id = []
    date = []
    team = []
    opp = []
    home = []
    team_score = []
    opp_score = []
    ot = []
    shootout = []
    non_shootout_win = []
    shootout_win = []
    for game in data:
        game_date = datetime.strptime(game[3], '%Y-%m-%d').date()
        if game_date >= curr_date:
            continue
        season.append(game[0])
        game_id.append(game[2])
        date.append(game_date)
        team.append(game[1])
        if game[4]["homeTeam"]["abbrev"] == game[1]:
            home.append(1)
            opp.append(game[4]["awayTeam"]["abbrev"])
            team_s = game[4]["homeTeam"]["score"]
            opp_s = game[4]["awayTeam"]["score"]
        else:
            home.append(0)
            opp.append(game[4]["homeTeam"]["abbrev"])
            team_s = game[4]["awayTeam"]["score"]
            opp_s = game[4]["homeTeam"]["score"]
        if game[4]["gameOutcome"]["lastPeriodType"] == "REG":
            ot.append(0)
            shootout.append(0)
            shootout_win.append(0)
            if team_s > opp_s:
                non_shootout_win.append(1)
            else:
                non_shootout_win.append(0)
        elif game[4]["gameOutcome"]["lastPeriodType"] == "SO":
            ot.append(0)
            shootout.append(1)
            non_shootout_win.append(0)
            if team_s > opp_s:
                shootout_win.append(1)
                team_s = team_s - 1
            else:
                shootout_win.append(0)
                opp_s = opp_s - 1
        elif game[4]["gameOutcome"]["lastPeriodType"] == "OT":
            ot.append(1)
            shootout.append(0)
            shootout_win.append(0)
            if team_s > opp_s:
                non_shootout_win.append(1)
            else:
                non_shootout_win.append(0)
        team_score.append(team_s)
        opp_score.append(opp_s)

    df = pd.DataFrame({
        "season": season,
        "game_id": game_id,
        "date": date,
        "team": team,
        "opp": opp,
        "home": home,
        "ot": ot,
        "shootout": shootout,
        "team_score": team_score,
        "opp_score": opp_score,
        "shootout_win": shootout_win,
        "non_shootout_win": non_shootout_win,
    })

    df['date'] = pd.to_datetime(df['date'])
    df['team_last_game_date'] = df['date']
    df['opp_last_game_date'] = df['date']
    df.sort_values(by=['opp', 'date'], inplace=True)
    last_team = None
    last_game_date = None

    for index, row in df.iterrows():
        if last_team == row['opp']:
            df.loc[index, 'opp_last_game_date'] = last_game_date
        else:
            df.loc[index, 'opp_last_game_date'] = None
        last_game_date = row['date']
        last_team = row['opp']

    df.sort_values(by=['team', 'date'], inplace=True)

    last_team = None
    last_game_date = None

    for index, row in df.iterrows():
        if last_team == row['team']:
            df.loc[index, 'team_last_game_date'] = last_game_date
        else:
            df.loc[index, 'team_last_game_date'] = None
        last_game_date = row['date']
        last_team = row['team']

    df['team_rest'] = (df['date'] - df['team_last_game_date']).dt.days - 1
    df['opp_rest'] = (df['date'] - df['opp_last_game_date']).dt.days - 1

    df.to_csv("storage/game_data.csv", index=False)


def curate_future_games(config, curr_date):
    data = config.get_boxscores()
    season = []
    game_id = []
    game_type = []
    date = []
    team = []
    opp = []
    home = []
    team_score = []
    opp_score = []
    ot = []
    shootout = []
    non_shootout_win = []
    shootout_win = []
    for game in data:
        game_date = datetime.strptime(game[3], '%Y-%m-%d').date()
        if game_date >= curr_date:
            future_game = 1
        else:
            future_game = 0
        game_type.append(future_game)
        season.append(game[0])
        game_id.append(game[2])
        date.append(game_date)
        team.append(game[1])
        if game[4]["homeTeam"]["abbrev"] == game[1]:
            home.append(1)
            opp.append(game[4]["awayTeam"]["abbrev"])
            if future_game == 1:
                team_s = None
                opp_s = None
            else:
                team_s = game[4]["homeTeam"]["score"]
                opp_s = game[4]["awayTeam"]["score"]
        else:
            home.append(0)
            opp.append(game[4]["homeTeam"]["abbrev"])
            if future_game == 1:
                team_s = None
                opp_s = None
            else:
                team_s = game[4]["awayTeam"]["score"]
                opp_s = game[4]["homeTeam"]["score"]
        if future_game == 1:
            ot.append(None)
            shootout.append(None)
            shootout_win.append(None)
            non_shootout_win.append(None)
        elif game[4]["gameOutcome"]["lastPeriodType"] == "REG":
            ot.append(0)
            shootout.append(0)
            shootout_win.append(0)
            if team_s > opp_s:
                non_shootout_win.append(1)
            else:
                non_shootout_win.append(0)
        elif game[4]["gameOutcome"]["lastPeriodType"] == "SO":
            ot.append(0)
            shootout.append(1)
            non_shootout_win.append(0)
            if team_s > opp_s:
                shootout_win.append(1)
                team_s = team_s - 1
            else:
                shootout_win.append(0)
                opp_s = opp_s - 1
        elif game[4]["gameOutcome"]["lastPeriodType"] == "OT":
            ot.append(1)
            shootout.append(0)
            shootout_win.append(0)
            if team_s > opp_s:
                non_shootout_win.append(1)
            else:
                non_shootout_win.append(0)
        team_score.append(team_s)
        opp_score.append(opp_s)

    df = pd.DataFrame({
        "season": season,
        "game_id": game_id,
        "future_game": game_type,
        "date": date,
        "team": team,
        "opp": opp,
        "home": home,
        "ot": ot,
        "shootout": shootout,
        "team_score": team_score,
        "opp_score": opp_score,
        "shootout_win": shootout_win,
        "non_shootout_win": non_shootout_win,
    })

    df['date'] = pd.to_datetime(df['date'])
    df['team_last_game_date'] = df['date']
    df['opp_last_game_date'] = df['date']
    df.sort_values(by=['opp', 'date'], inplace=True)
    last_team = None
    last_game_date = None

    for index, row in df.iterrows():
        if last_team == row['opp']:
            df.loc[index, 'opp_last_game_date'] = last_game_date
        else:
            df.loc[index, 'opp_last_game_date'] = None
        last_game_date = row['date']
        last_team = row['opp']

    df.sort_values(by=['team', 'date'], inplace=True)

    last_team = None
    last_game_date = None

    for index, row in df.iterrows():
        if last_team == row['team']:
            df.loc[index, 'team_last_game_date'] = last_game_date
        else:
            df.loc[index, 'team_last_game_date'] = None
        last_game_date = row['date']
        last_team = row['team']

    df['team_rest'] = (df['date'] - df['team_last_game_date']).dt.days - 1
    df['opp_rest'] = (df['date'] - df['opp_last_game_date']).dt.days - 1

    df = df[df['future_game'] == 1]
    df.to_csv("storage/future_games.csv", index=False)


