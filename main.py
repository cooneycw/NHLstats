from config.config import Config
from src_code.data_collect.collect_01 import get_season_data, get_team_list


def main():
    config = Config()
    get_season_data(config)
    get_team_list(config)


if __name__ == '__main__':
    main()
