import os
from dotenv import load_dotenv
import game_load
import stockfish_eval
import data_analysis

load_dotenv()
CHESS_USER = os.getenv("CHESS_USER")


def load_games(path):
    path = os.path.join(path, "games")
    if not os.path.isdir(path):
        os.mkdir(path)

    games_count = game_load.fetch_chess_games(CHESS_USER, path)
    print(f"Staženo {games_count} her uživatele {CHESS_USER}")
    return path


def analyze_games(games_path, path, stockfish_path):
    analyzed_path = os.path.join(path, "games_analyzed")
    if not os.path.isdir(analyzed_path):
        os.mkdir(analyzed_path)

    games_count = stockfish_eval.analyze_archives(games_path, analyzed_path, stockfish_path)
    print(f"Her analyzováno: {games_count}.")
    return analyzed_path


def main():
    if CHESS_USER:
        path = CHESS_USER + "_data"
        stockfish_path = "stockfish/stockfish-windows-x86-64-avx2.exe"
        if not os.path.isdir(path):
            os.mkdir(path)

        games_path = os.getenv("GAMES_PATH")
        if not os.path.isdir(games_path):
            games_path = load_games(path)

        analyzed_path = os.getenv("ANALYZED_PATH")
        if not os.path.isdir(analyzed_path):
            analyzed_path = analyze_games(games_path, path, stockfish_path)

    else:
        print("CHESS_USER není nastaven v .env souboru.")


if __name__ == "__main__":
    main()
