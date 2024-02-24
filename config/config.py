class Config:
    def __init__(self):
        self.base_url = "https://api-web.nhle.com"
        self.endpoints = {
            "seasons": "{base_url}/v1/season",
            "current_schedule": "/v1/schedule/now",
            "games": "",
            "schedule": "/v1/club-schedule-season/{team}/{season}"
        }
        self.study_period = 5
        self.seasons = None
        self.teams = None

    def set_seasons(self, seasons):
        self.seasons = seasons[-1*self.study_period:]
