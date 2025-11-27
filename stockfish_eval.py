import json
import io
import os
from multiprocessing import Pool, cpu_count
import chess
import chess.pgn
from stockfish import Stockfish
from tqdm import tqdm


def analyze_games(games_file, stockfish_path, output_file):
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

    print(f"Analýza {games_file} Dokončena.")
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
    stockfish.set_fen_position(board.fen())
    for node in game_pgn.mainline():
        move = node.move
        move_uci = move.uci()
        best_move_uci = stockfish.get_best_move()
        best_move_piece = get_piece_symbol(board, best_move_uci)
        moved_piece = get_piece_symbol(board, move_uci)
        is_best_moved_piece = chess.Move.from_uci(move_uci).from_square == chess.Move.from_uci(best_move_uci).from_square
        board.push(move)
        stockfish.set_fen_position(board.fen())
        stockfish_eval = stockfish.get_evaluation()

        pos_eval = transform_eval(stockfish_eval, as_white, board)

        moves.append({
            "move_order": len(moves) + 1,
            "pos_eval": pos_eval,
            "move_eval_change": pos_eval - last_eval,
            "is_mate": stockfish_eval["type"] == "mate",
            "mate_in_moves": abs(stockfish_eval["value"]) if stockfish_eval["type"] == "mate" else 0,
            "moved_piece": moved_piece,
            "best_move_piece": best_move_piece,
            "is_best_move": move.uci() == best_move_uci,
            "is_best_moved_piece": is_best_moved_piece,
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


