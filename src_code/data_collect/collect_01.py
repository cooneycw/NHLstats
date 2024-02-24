import requests


def get_season_data(config):
    # Endpoint to fetch player stats for a specific season
    url = f"{config.endpoints['seasons']}"
    final_url = url.format(base_url=config.base_url)
    response = requests.get(final_url)
    season_data = response.json()
    config.set_seasons(season_data)


def get_team_list(config):
    final_url = f"{config.endpoints['current_schedule']}"
    response = requests.get(final_url)
    schedule_data = response.json()
    cwc = 0