import os
from dotenv import load_dotenv
import game_load
import stockfish_eval

load_dotenv()
CHESS_USER = os.getenv("CHESS_USER")


def load_games(path):
    path = os.path.join(path, "games")
    if not os.path.isdir(path):
        os.mkdir(path)

    games_count = game_load.fetch_chess_games(CHESS_USER, path)
    print(f"Staženo {games_count} her uživatele {CHESS_USER}")
    return path


def eval_games(games_path, path, stockfish_path):
    path = os.path.join(path, "games_analysed")
    if not os.path.isdir(path):
        os.mkdir(path)

    games_count = 0
    files_done = 0
    for file in os.listdir(games_path):
        print(f"Souborů k analýze: {len(os.listdir(games_path))} Zbývá: {len(os.listdir(games_path)) - files_done}"
              f". Partií zpracováno: {games_count}.")
        if file.endswith(".json"):
            games_file = os.path.join(games_path, file)
            games_analyzed_file = os.path.join(path, f"analyzed_{file}")
            if os.path.isfile(games_analyzed_file):
                print(f"Soubor {games_analyzed_file} již existuje, přeskočeno.")
            else:
                games_count += stockfish_eval.analyze_games(games_file, stockfish_path=stockfish_path,
                                                            output_file=games_analyzed_file)
        files_done += 1

    print(f"Analýza {games_count} her dokončena.")


def main():
    if CHESS_USER:
        path = CHESS_USER + "_data"
        stockfish_path = "stockfish/stockfish-windows-x86-64-avx2.exe"
        if not os.path.isdir(path):
            os.mkdir(path)

        #  games_path = load_games(path)
        eval_games(path + "\games", path, stockfish_path)
    else:
        print("CHESS_USER není nastaven v .env souboru.")


if __name__ == "__main__":
    main()
