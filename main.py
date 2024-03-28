import asyncio
from datetime import datetime, timedelta
from config.config import Config
from src_code.data_collect.collect_01 import get_season_data, get_team_list, get_game_list, get_boxscore_list, get_rosters, get_player_list
from src_code.data_collect.collect_02 import get_lines
from src_code.data_curate.curate_01 import curate_basic_stats, curate_future_games
from src_code.data_curate.curate_02 import curate_rolling_stats, curate_proj_data
from src_code.data_curate.curate_03 import curate_player_stats, curate_future_player_stats
from src_code.data_curate.curate_04 import curate_rolling_player_stats, curate_proj_player_data
from src_code.data_analysis.analysis_01 import perform_tf, perform_logistic, perform_logistic_goals
from src_code.data_analysis.analysis_02 import start_h2o
from src_code.data_analysis.analysis_03 import perform_logistic_player, perform_kmeans_player
from src_code.data_analysis.analysis_04 import perform_tf_player
from src_code.utils.utils import save_data, load_data


def main():
    seg_list = [200]
    days_list = [10, 22]
    seasons = 6
    curr_date = datetime.now().date()
    config = Config(seasons)
    # get_data(config)
    get_lines(config)
    curate_data_seg(curr_date, seg_list)
    perform_kmeans_player()

    curate_data(curr_date, days_list)
    perform_tf()
    # perform_logistic()
    # perform_logistic_player(segs=True)
    perform_tf_player(segs=True)


def get_data(config):
    get_season_data(config)
    get_team_list(config)
    asyncio.run(get_game_list(config))
    asyncio.run(get_boxscore_list(config))
    asyncio.run(get_rosters(config))
    get_player_list(config)
    save_data(config)


def curate_data_seg(curr_date, seg_list):
    curate_data(curr_date, seg_list)


def curate_data(curr_date, days_list):
    config = load_data()
    curate_basic_stats(config, curr_date)
    curate_future_games(config, curr_date)
    curate_player_stats(config, curr_date)
    curate_future_player_stats(config, curr_date)
    for days in days_list:
        print(f"Game processing days: {days}")
        curate_rolling_stats(config, curr_date, days=days)
        curate_proj_data(config, curr_date, days=days)

    first_days = True
    for j, days in enumerate(days_list):
        if j != 0:
            first_days = False
        print(f"Player processing days: {days}")
        curate_rolling_player_stats(config, curr_date, first_days, days=days)
        curate_proj_player_data(config, curr_date, first_days, days=days)


if __name__ == '__main__':
    main()
