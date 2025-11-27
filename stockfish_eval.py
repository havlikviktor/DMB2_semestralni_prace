import json
import io
import os
import re
from multiprocessing import Pool, cpu_count
import chess
import chess.pgn
from stockfish import Stockfish
from tqdm import tqdm


def analyze_archives(archives_path, analyzed_path, stockfish_path):
    games_count = 0
    files_done = 0
    for archive in reversed(os.listdir(archives_path)):
        print(f"Souborů k analýze: {len(os.listdir(archives_path))} Zbývá: {len(os.listdir(archives_path)) - files_done}"
              f". Partií zpracováno: {games_count}.")
        if archive.endswith(".json"):
            archive_file = os.path.join(archives_path, archive)
            games_analyzed_file = os.path.join(analyzed_path, f"analyzed_{archive}")
            if os.path.isfile(games_analyzed_file):
                print(f"Soubor {games_analyzed_file} již analyzován, přeskočeno.")
            else:
                games_count += analyze_archive(archive_file, stockfish_path=stockfish_path, output_file=games_analyzed_file)
        files_done += 1
    return games_count


def analyze_archive(games_file, stockfish_path, output_file):
    print(f"Analyzuji archiv {games_file}")
    with open(games_file, "r", encoding="utf-8") as f:
        games_list = json.load(f)

    args_list = [(game, stockfish_path) for game in games_list]
    num_processes = min(int(os.getenv("ANALYSIS_PROCESSES_COUNT", 1)), cpu_count())

    with tqdm(total=len(args_list), desc="\033[92mAnalyzing games", unit="game") as pbar:
        with Pool(processes=num_processes) as pool:
            results = []
            for result in pool.imap_unordered(analyze_game, args_list):
                if result:
                    results.append(result)
                pbar.update(1)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return len(results)


def analyze_game(args):
    game, stockfish_path = args
    pgn = game.get("pgn", "")

    if not pgn:
        return None

    stockfish = Stockfish(path=stockfish_path, depth=15)
    stockfish.update_engine_parameters({"Threads": int(os.getenv("STOCKFISH_THREADS", 1))})
    game_pgn = chess.pgn.read_game(io.StringIO(pgn))
    board = game_pgn.board()
    moves_analyzed = eval_pgn(game_pgn, board, stockfish, game.get("as_white", True))

    del game["pgn"]
    game.update(analyze_castling(game.get("as_white", True), game_pgn))
    game["total_moves"] = board.fullmove_number
    game["games_analyzed"] = moves_analyzed
    return game


def eval_pgn(game_pgn, board, stockfish, as_white):
    moves = []
    last_eval = 0
    clock = {"w": None, "b": None}

    stockfish.set_fen_position(board.fen())
    for node in game_pgn.mainline():
        move = node.move
        move_uci = move.uci()
        best_move_uci = stockfish.get_best_move()
        best_move_piece = get_piece_symbol(board, best_move_uci)
        moved_piece = get_piece_symbol(board, move_uci)
        is_best_moved_piece = chess.Move.from_uci(move_uci).from_square == chess.Move.from_uci(best_move_uci).from_square
        pieces_on_board = len(board.piece_map())

        player = "w" if board.turn == chess.WHITE else "b"
        move_time = analyze_clock(node.comment, clock, player)

        board.push(move)
        stockfish.set_fen_position(board.fen())
        stockfish_eval = stockfish.get_evaluation()
        pos_eval = transform_eval(stockfish_eval, as_white, board)

        moves.append({
            "move_order": len(moves) + 1,
            "pos_eval": pos_eval,
            "move_eval_change": pos_eval - last_eval,
            "pieces_on_board": pieces_on_board,
            "mate_possible": stockfish_eval["type"] == "mate",
            "mate_in_moves": abs(stockfish_eval["value"]) if stockfish_eval["type"] == "mate" else 0,
            "moved_piece": moved_piece,
            "best_move_piece": best_move_piece,
            "is_best_move": move.uci() == best_move_uci,
            "is_best_moved_piece": is_best_moved_piece,
            "time_used_sec": move_time,
            "time_on_clock_sec": clock[player],
            "player": player,
        })
        last_eval = pos_eval

    return moves


def analyze_castling(as_white, game_pgn):
    meta = {
        "player_short_castle": False,
        "player_long_castle": False,
        "opponent_short_castle": False,
        "opponent_long_castle": False,
    }
    board = game_pgn.board()

    castle_map = {
        "e1g1": "short", "e8g8": "short",
        "e1c1": "long",  "e8c8": "long",
    }
    for move in game_pgn.mainline_moves():
        uci = move.uci()
        player_to_move = (board.turn == chess.WHITE) == as_white

        if uci in castle_map:
            key = ("player_" if player_to_move else "opponent_") + castle_map[uci] + "_castle"
            meta[key] = True
        board.push(move)
    return meta


def analyze_clock(comment, clock, player):
    clk = parse_clk(comment)
    if clk is not None:
        if clock[player] is not None:
            move_time = round(clock[player] - clk, 2)
        else:
            move_time = int(os.getenv("FETCH_TIME_CONTROL", 180)) - clk
    else:
        move_time = None
    if clk is not None:
        clock[player] = clk
    return move_time


def clk_to_seconds(clk_str):
    parts = clk_str.split(":")
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h = 0
        m, s = parts
    else:
        return None
    return int(h)*3600 + int(m)*60 + float(s)


def parse_clk(comment):
    if not comment:
        return None
    clk = re.search(r"\[%clk ([\d:.]+)]", comment)
    if clk:
        return clk_to_seconds(clk.group(1))
    return None


def transform_eval(eval_dict, as_white, board):
    sign = 1 if eval_dict["value"] > 0 else -1
    max_value = sign * (int(os.getenv("STOCKFISH_DEPTH", 10)) * 1000)

    if board.is_game_over():
        result = max_value if board.result() == "1-0" else -max_value if board.result() == "0-1" else 0
        return result if as_white else -result

    if eval_dict["type"] == "cp":
        pos_eval = eval_dict["value"]
    elif eval_dict["type"] == "mate":
        pos_eval = max_value / abs(eval_dict["value"])
    else:
        pos_eval = 0

    if not as_white:
        pos_eval = -pos_eval

    return pos_eval


def get_piece_symbol(board, move_uci):
    if move_uci:
        return board.piece_at(chess.Move.from_uci(move_uci).from_square).symbol().upper()
    return None
