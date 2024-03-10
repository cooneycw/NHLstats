from datetime import datetime
import pandas as pd
pd.set_option('display.max_columns', None)


def curate_player_stats(config):
    boxscore_data = config.get_boxscores()
    player_data = config.get_player_list()
    players = dict()
    for player in player_data:
        players[player[1]] = dict()
        players[player[1]]['team'] = player[0]
        players[player[1]]['position'] = player[2]
        players[player[1]]['lastName'] = player[3]['default']
        players[player[1]]['firstName'] = player[4]['default']

    player_id = []
    team = []
    season = []

    for boxscore in boxscore_data:
        t_season = boxscore[0]
        t_team = boxscore[1]
        t_game_id = boxscore[2]
        t_date = boxscore[3]
        data = boxscore[4]
        for teamType in data['boxscore']['playerByGameStats'].keys():
            sub_data = data['boxscore']['playerByGameStats'][teamType]
            for playerType in sub_data.keys():
                sub_sub_data = sub_data[playerType]
                for element in sub_sub_data:
                    cwc = 0

