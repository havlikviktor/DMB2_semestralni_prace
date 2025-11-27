import json
import io
import os
from multiprocessing import Pool, cpu_count
import chess
import chess.pgn
from stockfish import Stockfish
from tqdm import tqdm
from colorama import Fore, Style


def analyze_pgn(args):
    pgn, stockfish_path = args

    stockfish = Stockfish(path=stockfish_path, depth=15)
    stockfish.update_engine_parameters({"Threads": int(os.getenv("STOCKFISH_THREADS", 1))})

    game = chess.pgn.read_game(io.StringIO(pgn))
    board = game.board()

    for node in game.mainline():
        move = node.move
        board.push(move)
        stockfish.set_fen_position(board.fen())
        best_move = stockfish.get_best_move()
        eval_info = stockfish.get_evaluation()
        node.comment = f"eval={eval_info}, best_move={best_move}"

    return str(game)


def analyze_games(games_file, stockfish_path, output_file):
    with open(games_file, "r", encoding="utf-8") as f:
        games_list = json.load(f)

    args_list = [(game.get("pgn", ""), stockfish_path) for game in games_list]

    num_processes = min(int(os.getenv("ANALYSIS_PROCESSES_COUNT", 1)), cpu_count())

    with tqdm(total=len(args_list), desc="\033[92mAnalyzing games", unit="game") as pbar:
        with Pool(processes=num_processes) as pool:
            results = []
            for result in pool.imap_unordered(analyze_pgn, args_list):
                results.append(result)
                pbar.update(1)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Analýza {games_file} Dokončena.")
    return len(results)
