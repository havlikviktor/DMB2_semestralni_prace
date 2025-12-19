import json
import os
from scipy.stats import pearsonr
import pandas as pd
from tqdm import tqdm


def load_chess_data(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ------- GAME LEVEL DATA -------
    df_games = pd.DataFrame([
        {
            "game_url": g["game_url"],
            "eco": g["eco"],
            "day_hour": g["day_hour"],
            "as_white": g["as_white"],
            "player_rating": g["player_rating"],
            "player_result": g["player_result"],
            "opponent_rating": g["opponent_rating"],
            "opponent_result": g["opponent_result"],
            "opponent_country": g["opponent_country"],
            "player_short_castle": g["player_short_castle"],
            "player_long_castle": g["player_long_castle"],
            "opponent_short_castle": g["opponent_short_castle"],
            "opponent_long_castle": g["opponent_long_castle"],
            "total_moves": g["total_moves"]
        }
        for g in data
    ])

    # ------- MOVE LEVEL DATA -------
    move_rows = []
    for game in data:
        for move in game["moves_analyzed"]:
            move_rows.append({
                "game_url": game["game_url"],  # KEY pro join
                **move
            })

    df_moves = pd.DataFrame(move_rows)

    return df_games, df_moves


def separate_player_moves(df_games, df_moves):
    # join pro info jestli hráč hrál bílými v dané partii
    df = df_moves.merge(df_games[["game_url", "as_white"]], on="game_url")

    df_user_moves = df[
        ((df["as_white"] == True) & (df["player"] == "w")) |
        ((df["as_white"] == False) & (df["player"] == "b")) &
        (df["move_eval_change"] < 1000)
        ].copy()

    df_opponent_moves = df[
        ((df["as_white"] == True) & (df["player"] == "b")) |
        ((df["as_white"] == False) & (df["player"] == "w"))
        ].copy()

    return df_user_moves, df_opponent_moves


def run_hypothesis_tests(df_games: pd.DataFrame, df_moves: pd.DataFrame):
    df_player_moves, df_opponent_moves = separate_player_moves(df_games, df_moves)

    print("\n===== HYPOTÉZA 1: Vyšší rating hráče → menší chyba =====")
    df_games["win"] = df_games["player_result"].apply(lambda x: True if x.lower() == "win" else False)
    df_player_moves["error"] = df_player_moves["move_eval_change"].apply(lambda x: -x if x < 0 else 0)
    df = df_player_moves.merge(df_games[["game_url", "player_rating"]], on="game_url")

    corr, p = pearsonr(df["player_rating"], df["error"])
    print(f"Korelace rating ↔ chyba: {corr:.4f} (p={p:.4f})")

    print("\n===== HYPOTÉZA 3: Hráč má lepší eval jako bílý než jako černý =====")

    df_games["short"] = df_games["total_moves"].apply(lambda x: 1 if x < 30 else 0)
    corr, p = pearsonr(df_games["win"], df_games["short"])
    print(f"Korelace win <-> short {corr} (p={p})")


def data_analysis(data_path):
    df_games, df_moves = pd.DataFrame(), pd.DataFrame()
    if not os.path.isdir(data_path):
        print(f"Adresář {data_path} neexistuje. Nejdříve spusťte analýzu her.")
    elif len(os.listdir(data_path)) == 0:
        print(f"Adresář {data_path} je prázdný. Nejdříve spusťte analýzu her.")
    else:
        for file in tqdm(reversed(os.listdir(data_path)), total=len(os.listdir(data_path)),
                         desc="\033[92mNačítání dat k analýze"):
            path = os.path.join(data_path, file)
            df_games_file, df_moves_file = load_chess_data(path)
            df_games = pd.concat([df_games, df_games_file], ignore_index=True)
            df_moves = pd.concat([df_moves, df_moves_file], ignore_index=True)

            run_hypothesis_tests(df_games, df_moves)
