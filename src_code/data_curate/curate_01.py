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
    t_score = []
    o_score = []
    t_sog = []
    o_sog = []
    t_hits = []
    o_hits = []
    t_blocks = []
    o_blocks = []
    t_pim = []
    o_pim = []
    t_face = []
    o_face = []

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
            team_score = game[4]["homeTeam"]["score"]
            opp_score = game[4]["awayTeam"]["score"]
            for category in game[4]['boxscore']['teamGameStats']:
                if category['category'] == 'sog':
                    team_sog = category['homeValue']
                    opp_sog = category['awayValue']
                elif category['category'] == 'hits':
                    team_hits = category['homeValue']
                    opp_hits = category['awayValue']
                elif category['category'] == 'blockedShots':
                    team_blocks = category['homeValue']
                    opp_blocks = category['awayValue']
                elif category['category'] == 'pim':
                    team_pim = category['homeValue']
                    opp_pim = category['awayValue']
                elif category['category'] == 'faceoffWinningPctg':
                    team_face = category['homeValue']
                    opp_face = category['awayValue']
        else:
            home.append(0)
            opp.append(game[4]["homeTeam"]["abbrev"])
            team_score = game[4]["awayTeam"]["score"]
            opp_score = game[4]["homeTeam"]["score"]
            for category in game[4]['boxscore']['teamGameStats']:
                if category['category'] == 'sog':
                    team_sog = category['awayValue']
                    opp_sog = category['homeValue']
                elif category['category'] == 'hits':
                    team_hits = category['awayValue']
                    opp_hits = category['homeValue']
                elif category['category'] == 'blockedShots':
                    team_blocks = category['awayValue']
                    opp_blocks = category['homeValue']
                elif category['category'] == 'pim':
                    team_pim = category['awayValue']
                    opp_pim = category['homeValue']
                elif category['category'] == 'faceoffWinningPctg':
                    team_face = category['awayValue']
                    opp_face = category['homeValue']
        if game[4]["gameOutcome"]["lastPeriodType"] == "REG":
            ot.append(0)
            shootout.append(0)
            shootout_win.append(0)
            if team_score > opp_score:
                non_shootout_win.append(1)
            else:
                non_shootout_win.append(0)
        elif game[4]["gameOutcome"]["lastPeriodType"] == "SO":
            ot.append(0)
            shootout.append(1)
            non_shootout_win.append(0)
            if team_score > opp_score:
                shootout_win.append(1)
                team_score = team_score - 1
            else:
                shootout_win.append(0)
                opp_score = opp_score - 1
        elif game[4]["gameOutcome"]["lastPeriodType"] == "OT":
            ot.append(1)
            shootout.append(0)
            shootout_win.append(0)
            if team_score > opp_score:
                non_shootout_win.append(1)
            else:
                non_shootout_win.append(0)
        t_score.append(team_score)
        o_score.append(opp_score)
        t_sog.append(team_sog)
        o_sog.append(opp_sog)
        t_hits.append(team_hits)
        o_hits.append(opp_hits)
        t_blocks.append(team_blocks)
        o_blocks.append(opp_blocks)
        t_pim.append(team_pim)
        o_pim.append(opp_pim)
        t_face.append(team_face)
        o_face.append(opp_face)

    df = pd.DataFrame({
        "season": season,
        "game_id": game_id,
        "date": date,
        "team": team,
        "opp": opp,
        "home": home,
        "ot": ot,
        "shootout": shootout,
        "team_score": t_score,
        "opp_score": o_score,
        "team_sog": t_sog,
        "opp_sog": o_sog,
        "team_hits": t_hits,
        "opp_hits": o_hits,
        "team_blocks": t_blocks,
        "opp_blocks": o_blocks,
        "team_pim": t_pim,
        "opp_pim": o_pim,
        "team_face": t_face,
        "opp_face": o_face,
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
    t_score = []
    o_score = []
    t_sog = []
    o_sog = []
    t_hits = []
    o_hits = []
    t_blocks = []
    o_blocks = []
    t_pim = []
    o_pim = []
    t_face = []
    o_face = []
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
                team_score = None
                opp_score = None
                team_sog = None
                opp_sog = None
                team_hits = None
                opp_hits = None
                team_blocks = None
                opp_blocks = None
                team_pim = None
                opp_pim = None
                team_face = None
                opp_face = None
            else:
                team_score = game[4]["homeTeam"]["score"]
                opp_score = game[4]["awayTeam"]["score"]
                for category in game[4]['boxscore']['teamGameStats']:
                    if category['category'] == 'sog':
                        team_sog = category['homeValue']
                        opp_sog = category['awayValue']
                    elif category['category'] == 'hits':
                        team_hits = category['homeValue']
                        opp_hits = category['awayValue']
                    elif category['category'] == 'blockedShots':
                        team_blocks = category['homeValue']
                        opp_blocks = category['awayValue']
                    elif category['category'] == 'pim':
                        team_pim = category['homeValue']
                        opp_pim = category['awayValue']
                    elif category['category'] == 'faceoffWinningPctg':
                        team_face = category['homeValue']
                        opp_face = category['awayValue']
        else:
            home.append(0)
            opp.append(game[4]["homeTeam"]["abbrev"])
            if future_game == 1:
                team_score = None
                opp_score = None
                team_sog = None
                opp_sog = None
                team_hits = None
                opp_hits = None
                team_blocks = None
                opp_blocks = None
                team_pim = None
                opp_pim = None
                team_face = None
                opp_face = None
            else:
                team_score = game[4]["awayTeam"]["score"]
                opp_score = game[4]["homeTeam"]["score"]
                for category in game[4]['boxscore']['teamGameStats']:
                    if category['category'] == 'sog':
                        team_sog = category['awayValue']
                        opp_sog = category['homeValue']
                    elif category['category'] == 'hits':
                        team_hits = category['awayValue']
                        opp_hits = category['homeValue']
                    elif category['category'] == 'blockedShots':
                        team_blocks = category['awayValue']
                        opp_blocks = category['homeValue']
                    elif category['category'] == 'pim':
                        team_pim = category['awayValue']
                        opp_pim = category['homeValue']
                    elif category['category'] == 'faceoffWinningPctg':
                        team_face = category['awayValue']
                        opp_face = category['homeValue']
        if future_game == 1:
            ot.append(None)
            shootout.append(None)
            shootout_win.append(None)
            non_shootout_win.append(None)
        elif game[4]["gameOutcome"]["lastPeriodType"] == "REG":
            ot.append(0)
            shootout.append(0)
            shootout_win.append(0)
            if team_score > opp_score:
                non_shootout_win.append(1)
            else:
                non_shootout_win.append(0)
        elif game[4]["gameOutcome"]["lastPeriodType"] == "SO":
            ot.append(0)
            shootout.append(1)
            non_shootout_win.append(0)
            if team_score > opp_score:
                shootout_win.append(1)
                team_score = team_score - 1
            else:
                shootout_win.append(0)
                opp_score = opp_score - 1
        elif game[4]["gameOutcome"]["lastPeriodType"] == "OT":
            ot.append(1)
            shootout.append(0)
            shootout_win.append(0)
            if team_score > opp_score:
                non_shootout_win.append(1)
            else:
                non_shootout_win.append(0)
        t_score.append(team_score)
        o_score.append(opp_score)
        t_sog.append(team_sog)
        o_sog.append(opp_sog)
        t_hits.append(team_hits)
        o_hits.append(opp_hits)
        t_blocks.append(team_blocks)
        o_blocks.append(opp_blocks)
        t_pim.append(team_pim)
        o_pim.append(opp_pim)
        t_face.append(team_face)
        o_face.append(opp_face)

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
        "team_score": t_score,
        "opp_score": o_score,
        "team_sog": t_sog,
        "opp_sog": o_sog,
        "team_hits": t_hits,
        "opp_hits": o_hits,
        "team_blocks": t_blocks,
        "opp_blocks": o_blocks,
        "team_pim": t_pim,
        "opp_pim": o_pim,
        "team_face": t_face,
        "opp_face": o_face,
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


