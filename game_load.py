import os
from datetime import datetime
from multiprocessing import Pool

import requests
import json

from tqdm import tqdm


def fetch_chess_games(username, path):
    games_count = 0
    url = f"https://api.chess.com/pub/player/{username}/games/archives"
    headers = {"User-Agent": "MyChessStats/1.0"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        archives = response.json().get("archives", [])
        for archive_url in reversed(archives):
            archive_date = archive_url.split("/")
            archive_path = os.path.join(path, f"{archive_date[-2]}_{archive_date[-1]}_blitz_180.json")

            if os.path.isfile(archive_path):
                print(f"Archiv {archive_url} již existuje, přeskočeno.")
            else:
                games_count += proces_archive(archive_url, archive_path, username)

    else:
        print(f"Chyba při načítání archivů. Status: {response.status_code}")

    return games_count


def proces_archive(archive_url, archive_path, username):
    print(f"Načítám archiv {archive_url}")
    games = []

    headers = {"User-Agent": "MyChessStats/1.0"}
    archive_response = requests.get(archive_url, headers=headers)

    if archive_response.status_code == 200:
        games_list = archive_response.json().get("games", [])
        args_list = [(game, username) for game in games_list]

        with Pool(processes=int(os.getenv("FETCH_PROCESSES_COUNT", 1))) as pool:
            for game_result in tqdm(pool.imap_unordered(process_game, args_list),
                                    total=len(args_list), desc="\033[92mProcessing games"):
                if game_result:
                    games.append(game_result)
    else:
        print(f"Chyba při načítání: {archive_path} archivu. Status: {archive_response.status_code}")

    save_games_to_file(games, archive_path)
    return len(games)


def process_game(args: (dict, str)):
    game, username = args
    tc = game.get("time_control", "")
    time_class = game.get("time_class", "")

    if time_class == "blitz" and tc == "180":
        eco = get_eco_code(game.get("pgn", ""))
        pgn = strip_pgn_tags(game.get("pgn", ""))
        day_hour = datetime.fromtimestamp(game.get("end_time", 0)).hour
        white = game.get("white", {})
        black = game.get("black", {})

        as_white = white.get("username", "").lower() == username.lower()
        opponent_country = get_opponent_country(black if as_white else white)
        if as_white:
            player_rating = white.get("rating", 0)
            player_result = white.get("result", "")
            opponent_rating = black.get("rating", 0)
            opponent_result = black.get("result", "")
        else:
            player_rating = black.get("rating", 0)
            player_result = black.get("result", "")
            opponent_rating = white.get("rating", 0)
            opponent_result = white.get("result", "")

        return {

            "pgn": pgn,
            "eco": eco,
            "day_hour": day_hour,
            "as_white": as_white,
            "player_rating": player_rating,
            "player_result": player_result,
            "opponent_rating": opponent_rating,
            "opponent_result": opponent_result,
            "opponent_country": opponent_country
        }
    return None


def save_games_to_file(games, filename):
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(games, file, ensure_ascii=False, indent=4)


def strip_pgn_tags(pgn_string: str) -> str:
    parts = pgn_string.split("\n\n", 1)
    if len(parts) == 2:
        return parts[1].strip()
    else:
        return pgn_string.strip()


def get_eco_code(pgn_string: str) -> str:
    for line in pgn_string.splitlines():
        if line.startswith("[ECO "):
            return line.split('"')[1]
    return "Unknown"


def get_opponent_country(opponent):
    opponent_url = opponent.get("@id", "")
    headers = {"User-Agent": "MyChessStats/1.0"}

    opponent_response = requests.get(opponent_url, headers=headers)
    if opponent_response.status_code == 200:
        country_url = opponent_response.json().get("country", "")
        country_response = requests.get(country_url, headers=headers)
        if country_response.status_code == 200:
            return country_response.json().get("name", "Unknown")
        else:
            print(f"Chyba při načítání země: {country_url}. Status: {country_response.status_code}")
    else:
        print(f"Chyba při načítání oponenta: {opponent_url}. Status: {opponent_url.status_code}")
    return "Unknown"
