class Config:
    def __init__(self, seasons):
        self.base_url = "https://api-web.nhle.com"
        self.endpoints = {
            "seasons": "{base_url}/v1/season",
            "standings": "{base_url}/v1/standings/now",
            "schedule": "{base_url}/v1/club-schedule-season/{team}/{season}",
            "boxscore": "{base_url}/v1/gamecenter/{game_id}/boxscore",
            "roster": "{base_url}/v1/roster/{team}/current",
        }
        self.study_period = seasons
        self.seasons = None
        self.teams = None
        self.games = None
        self.boxscores = None
        self.rosters = None
        self.player_list = None

    def set_seasons(self, seasons):
        self.seasons = seasons[-1*self.study_period:]

    def set_teams(self, teams):
        self.teams = teams

    def set_games(self, games):
        self.games = games

    def set_boxscores(self, boxscores):
        self.boxscores = boxscores

    def set_rosters(self, rosters):
        self.rosters = rosters

    def set_player_list(self, player_list):
        self.player_list = player_list

    def get_seasons(self):
        return self.seasons

    def get_teams(self):
        return self.teams

    def get_games(self):
        return self.games

    def get_boxscores(self):
        return self.boxscores

    def get_rosters(self):
        return self.rosters

    def get_player_list(self):
        return self.player_list
