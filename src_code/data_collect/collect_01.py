import requests
import aiohttp
import asyncio


def get_season_data(config):
    # Endpoint to fetch player stats for a specific season
    url = f"{config.endpoints['seasons']}"
    final_url = url.format(base_url=config.base_url)
    response = requests.get(final_url)
    season_data = response.json()
    config.set_seasons(season_data)


def get_team_list(config):
    url = f"{config.endpoints['standings']}"
    final_url = url.format(base_url=config.base_url)
    response = requests.get(final_url)
    standings_data = response.json()
    team_list = []
    for item in standings_data['standings']:
        team_abbrev = item['teamAbbrev']['default']
        team_list.append(team_abbrev)
    team_list.sort()
    config.set_teams(team_list)


async def get_rosters(config):
    tasks = []
    async with aiohttp.ClientSession() as session:
        for team in config.get_teams():
            url = f"{config.endpoints['roster']}"
            final_url = url.format(base_url=config.base_url, team=team)
            task = asyncio.create_task(fetch_roster_data(session, final_url, team))
            tasks.append(task)
        roster_list = await asyncio.gather(*tasks)
    config.set_rosters(roster_list)


def get_player_list(config):
    roster_data = config.get_rosters()
    player_list = []
    for team, roster_info in roster_data:
        for player_type in roster_info.keys():
            for player in roster_info[player_type]:
                identity = (team, player['id'], player_type, player['lastName'], player['firstName'])
                player_list.append(identity)
    config.set_player_list(player_list)


async def get_game_list(config):
    tasks = []
    async with aiohttp.ClientSession() as session:
        for season in config.get_seasons():
            for team in config.get_teams():
                final_url = config.endpoints['schedule'].format(base_url=config.base_url, team=team, season=season)
                task = asyncio.create_task(fetch_game_data(session, final_url, season, team))
                tasks.append(task)
        game_list = await asyncio.gather(*tasks)
        # Flatten the list of lists
        game_list = [game for sublist in game_list for game in sublist]
    config.set_games(game_list)


async def get_boxscore_list(config):
    boxscore_list = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        for game_id in config.get_games():
            final_url = config.endpoints['boxscore'].format(base_url=config.base_url, game_id=game_id[2])
            task = asyncio.create_task(fetch_boxscore_data(session, final_url, game_id))
            tasks.append(task)
        results = await asyncio.gather(*tasks)
        boxscore_list.extend(results)
    config.set_boxscores(boxscore_list)


async def fetch_roster_data(session, url, team):
    async with session.get(url) as response:
        roster_data = await response.json()
        return (team, roster_data)


async def fetch_game_data(session, url, season, team):
    async with session.get(url) as response:
        schedule_data = await response.json()
        return [(season, team, game['id'], game['gameDate']) for game in schedule_data['games'] if game['gameType'] == 2]


async def fetch_boxscore_data(session, url, game_id):
    async with session.get(url) as response:
        boxscore_data = await response.json()
        return (game_id[0], game_id[1], game_id[2], game_id[3], boxscore_data)
