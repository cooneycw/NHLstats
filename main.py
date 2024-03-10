import asyncio
from datetime import datetime
from config.config import Config
from src_code.data_collect.collect_01 import get_season_data, get_team_list, get_game_list, get_boxscore_list, get_rosters, get_player_list
from src_code.data_curate.curate_01 import curate_basic_stats, curate_future_games
from src_code.data_curate.curate_02 import curate_rolling_stats, curate_proj_data
from src_code.data_curate.curate_03 import curate_player_stats
from src_code.data_analysis.analysis_01 import perform_tf, perform_logistic, perform_logistic_goals
from src_code.data_analysis.analysis_02 import start_h2o
from src_code.utils.utils import save_data, load_data


def main():
    days_list = [10, 22]
    goal_test_list = [4, 5, 6]
    seasons = 5
    curr_date = datetime.now().date()
    # get_data(seasons)
    curate_data(curr_date, days_list)
    # perform_tf()
    # perform_logistic()
    # for goal_test in goal_test_list:
    #    perform_logistic_goals(goal_test)
    # start_h2o()


def get_data(seasons):
    config = Config(seasons)
    get_season_data(config)
    get_team_list(config)
    asyncio.run(get_game_list(config))
    asyncio.run(get_boxscore_list(config))
    asyncio.run(get_rosters(config))
    get_player_list(config)
    save_data(config)


def curate_data(curr_date, days_list):
    config = load_data()
    # curate_basic_stats(config, curr_date)
    # curate_future_games(config, curr_date)
    # for days in days_list:
    #     print(f"processing days: {days}")
    #     curate_rolling_stats(config, curr_date, days=days)
    #     curate_proj_data(config, curr_date, days=days)
    curate_player_stats(config)


if __name__ == '__main__':
    main()
