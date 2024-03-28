import random
import requests
import time
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup


def get_lines(config):
    url = f"{config.endpoints['lines']}"
    teams = [
        ('ANA', 'anaheim-ducks'),
        ('ARI', 'arizona-coyotes'),
        ('BOS', 'boston-bruins'),
        ('BUF', 'buffalo-sabres'),
        ('CGY', 'calgary-flames'),
        ('CAR', 'carolina-hurricanes'),
        ('CHI', 'chicago-blackhawks'),
        ('COL', 'colorado-avalanche'),
        ('CBJ', 'columbus-blue-jackets'),
        ('DAL', 'dallas-stars'),
        ('DET', 'detroit-red-wings'),
        ('EDM', 'edmonton-oilers'),
        ('FLA', 'florida-panthers'),
        ('LAK', 'los-angeles-kings'),
        ('LAK', 'los-angeles-kings'),
        ('MIN', 'minnesota-wild'),
        ('MTL', 'montreal-canadiens'),
        ('NSH', 'nashville-predators'),
        ('NJD', 'new-jersey-devils'),
        ('NYI', 'new-york-islanders'),
        ('NYR', 'new-york-rangers'),
        ('OTT', 'ottawa-senators'),
        ('PHI', 'philadelphia-flyers'),
        ('PIT', 'pittsburgh-penguins'),
        ('SJS', 'san-jose-sharks'),
        ('STL', 'st-louis-blues'),
        ('TBL', 'tampa-bay-lightning'),
        ('TOR', 'toronto-maple-leafs'),
        ('VAN', 'vancouver-canucks'),
        ('VGK', 'vegas-golden-knights'),
        ('WSH', 'washington-capitals'),
        ('WPG', 'winnipeg-jets'),
             ]

    playerTeam = []
    playerTime = []
    playerInd = []
    playerLine = []
    playerLastName = []
    playerFirstName = []
    positionCentre = []
    positionRightWing = []
    positionLeftWing = []
    positionDefense = []
    positionGoalie = []

    for team in teams:
        final_url = url.format(base_url_lines=config.base_url_lines, line_team=team[1])
        response = requests.get(final_url, headers=config.headers_lines)
        soup = BeautifulSoup(response.content, 'html.parser')

        span_element = soup.find(text="Last updated:")

        updated_time = None
        if span_element:
            next_span = span_element.find_next('span')
            if next_span:
                updated_time = datetime.strptime(next_span.text.strip(), '%Y-%m-%dT%H:%M:%S.%fZ')


        forwards_span = soup.find('span', id='forwards', class_='text-3xl text-white')
        images_after_forwards = forwards_span.find_all_next('img')
        line = 0
        for i, image in enumerate(images_after_forwards):
            if i > 11:
                break
            else:
                player_name = image['alt']
                p_FirstName, p_LastName = player_name.split(maxsplit=1)
                playerTeam.append(team[0])
                playerTime.append(updated_time)
                playerInd.append(i)
                playerLastName.append(p_LastName)
                playerFirstName.append(p_FirstName)
                playerLine.append(line)
            if i % 3 == 0 and i <= 11:
                positionCentre.append(0)
                positionRightWing.append(0)
                positionLeftWing.append(1)
                positionDefense.append(0)
                positionGoalie.append(0)
            elif i % 3 == 1 and i <= 11:
                positionCentre.append(1)
                positionRightWing.append(0)
                positionLeftWing.append(0)
                positionDefense.append(0)
                positionGoalie.append(0)
            elif i % 3 == 2 and i <= 11:
                positionCentre.append(0)
                positionRightWing.append(1)
                positionLeftWing.append(0)
                positionDefense.append(0)
                positionGoalie.append(0)
                line += 1

        defense_span = soup.find('span', id='defense', class_='text-3xl text-white')
        images_after_defense = defense_span.find_all_next('img')

        for i, image in enumerate(images_after_defense):
            if i > 5:
                break
            else:
                player_name = image['alt']
                p_FirstName, p_LastName = player_name.split(maxsplit=1)
                playerTeam.append(team[0])
                playerTime.append(updated_time)
                playerInd.append(i+12)
                playerLastName.append(p_LastName)
                playerFirstName.append(p_FirstName)
                playerLine.append(line)
                positionCentre.append(0)
                positionRightWing.append(0)
                positionLeftWing.append(0)
                positionDefense.append(1)
                positionGoalie.append(0)
                if i % 2 == 1:
                    line += 1

        goalie_span = soup.find('span', id='goalies', class_='text-3xl text-white')
        images_after_goalie = goalie_span.find_all_next('img')

        line = 0
        for i, image in enumerate(images_after_goalie):
            if i > 1:
                break
            else:
                player_name = image['alt']
                p_FirstName, p_LastName = player_name.split(maxsplit=1)
                playerTeam.append(team[0])
                playerTime.append(updated_time)
                playerInd.append(i+18)
                playerLastName.append(p_LastName)
                playerFirstName.append(p_FirstName)
                playerLine.append(line)
                positionCentre.append(0)
                positionRightWing.append(0)
                positionLeftWing.append(0)
                positionDefense.append(0)
                positionGoalie.append(1)
                line += 1

        time.sleep(random.uniform(8, 35))

    current_time = datetime.now()
    formatted_date = "{:04d}_{:02d}_{:02d}".format(current_time.year, current_time.month, current_time.day)

    current_lines_df = pd.DataFrame({
        "team": playerTeam,
        "updateTime": playerTime,
        "playerInd": playerInd,
        "playerLine": playerLine,
        "playerLastName": playerLastName,
        "playerFirstName": playerFirstName,
        "positionCentre": positionCentre,
        "positionRightWing": positionRightWing,
        "positionLeftWing": positionLeftWing,
        "positionDefense": positionDefense,
        "positionGoalie": positionGoalie,
    })

    current_lines_df.to_csv(f"storage/current_lines_{formatted_date}.csv", index=False)

