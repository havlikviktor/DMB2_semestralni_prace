import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Set plot style
sns.set_theme(style="whitegrid")

def load_all_data(analyzed_path):
    """
    Loads all JSON files from the analyzed_path directory and converts them 
    into two DataFrames: df_games and df_moves.
    """
    all_games = []
    all_moves = []
    
    json_files = glob.glob(os.path.join(analyzed_path, "*.json"))
    
    print(f"Loading data from {len(json_files)} files...")
    
    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                games = json.load(f)
                for game in games:
                    # Extract game level info
                    game_info = {
                        "game_url": game.get("game_url"),
                        "eco": game.get("eco"),
                        "day_hour": game.get("day_hour"),
                        "as_white": game.get("as_white"),
                        "player_rating": game.get("player_rating"),
                        "opponent_rating": game.get("opponent_rating"),
                        "player_result": game.get("player_result"),
                        "opponent_result": game.get("opponent_result"),
                        "opponent_country": game.get("opponent_country"),
                        "player_short_castle": game.get("player_short_castle"),
                        "player_long_castle": game.get("player_long_castle"),
                        "opponent_short_castle": game.get("opponent_short_castle"),
                        "opponent_long_castle": game.get("opponent_long_castle"),
                        "total_moves": game.get("total_moves"),
                    }
                    all_games.append(game_info)
                    
                    # Extract moves
                    moves = game.get("moves_analyzed", [])
                    for move in moves:
                        move_data = move.copy()
                        move_data["game_url"] = game.get("game_url")
                        # Add a flag for castling matching
                        # (Not strictly move-level but useful context if needed, though handled in game analysis)
                        all_moves.append(move_data)
                        
            except json.JSONDecodeError:
                print(f"Error decoding {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    df_games = pd.DataFrame(all_games)
    df_moves = pd.DataFrame(all_moves)
    
    # Post-processing df_games
    # Add a binary 'win' column
    df_games['is_win'] = df_games['player_result'] == 'win'
    
    print(f"Loaded {len(df_games)} games and {len(df_moves)} moves.")
    return df_games, df_moves

def plot_winrate_overall(df_games):
    """Calculates and plots the overall win rate."""
    win_counts = df_games['player_result'].value_counts(normalize=True) * 100
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=win_counts.index, y=win_counts.values, hue=win_counts.index, palette="viridis", legend=False)
    plt.xticks(rotation=45, ha='right')
    plt.title("Overall Game Results")
    plt.ylabel("Percentage (%)")
    plt.xlabel("Result")
    for i, v in enumerate(win_counts.values):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')
    plt.tight_layout()
    plt.show()

def plot_country_winrate(df_games, min_games=5):
    """Plots win rate by opponent's country (for countries with at least min_games)."""
    country_stats = df_games.groupby('opponent_country').agg(
        games=('game_url', 'count'),
        wins=('is_win', 'sum')
    )
    country_stats['win_rate'] = (country_stats['wins'] / country_stats['games']) * 100
    
    # Filter for countries with enough games
    country_stats = country_stats[country_stats['games'] >= min_games].sort_values('win_rate', ascending=False)
    
    if country_stats.empty:
        print(f"No countries with at least {min_games} games found.")
        return

    plt.figure(figsize=(12, 6))
    sns.barplot(x=country_stats.index, y=country_stats['win_rate'], hue=country_stats.index, palette="coolwarm", legend=False)
    plt.xticks(rotation=90)
    plt.title(f"Win Rate by Opponent Country (min {min_games} games)")
    plt.ylabel("Win Rate (%)")
    plt.xlabel("Country")
    plt.tight_layout()
    plt.show()

def classify_phase(row):
    """Classifies the game phase based on pieces on board and move number."""
    # Heuristic: 
    # Opening: Move count <= 10
    # Endgame: Pieces <= 14 (Kings + 6 pieces each approx)
    # Middlegame: The rest
    if row['move_order'] <= 10:
        return 'Opening'
    elif row['pieces_on_board'] <= 14:
        return 'Endgame'
    else:
        return 'Middlegame'

def analyze_game_phases(df_moves):
    """Analyzes performance and critical pieces in Opening, Middlegame, and Endgame."""
    df_moves['phase'] = df_moves.apply(classify_phase, axis=1)
    
    # Filter for player's moves only for accuracy analysis
    player_moves = df_moves[df_moves['players_move'] == True].copy()
    
    # Accuracy: measure by move_eval_change (closer to 0 or positive is better)
    # Note: Stockfish eval change is (Current - Previous). If I make a move, 
    # optimal is usually 0 change (maintaining advantage). Negative means blunder/mistake.
    
    phase_stats = player_moves.groupby('phase')['move_eval_change'].mean().reset_index()
    phase_stats_accuracy = player_moves.groupby('phase')['is_best_move'].mean().reset_index() # % of best moves
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Average Eval Change (Quality)
    sns.barplot(data=phase_stats, x='phase', y='move_eval_change', ax=axes[0], order=['Opening', 'Middlegame', 'Endgame'], hue='phase', palette='magma', legend=False)
    axes[0].set_title("Average Eval Change by Phase (Higher is better)")
    axes[0].set_ylabel("Avg Centipawn Change")
    
    # Plot Accuracy (% Best Moves)
    sns.barplot(data=phase_stats_accuracy, x='phase', y='is_best_move', ax=axes[1], order=['Opening', 'Middlegame', 'Endgame'], hue='phase', palette='viridis', legend=False)
    axes[1].set_title("Move Accuracy (% Best Move) by Phase")
    axes[1].set_ylabel("Accuracy (0-1)")
    
    plt.tight_layout()
    plt.show()
    
    # Critical Piece Analysis
    print("\n--- Critical Pieces by Phase (Worst Avg Eval Change) ---")
    for phase in ['Opening', 'Middlegame', 'Endgame']:
        phase_data = player_moves[player_moves['phase'] == phase]
        if phase_data.empty:
            continue
        piece_stats = phase_data.groupby('moved_piece')['move_eval_change'].mean().sort_values()
        print(f"\n{phase}:")
        print(piece_stats.head(3)) # Show top 3 worst performing pieces
        
        # Plot for each phase
        plt.figure(figsize=(8, 5))
        sns.barplot(x=piece_stats.index, y=piece_stats.values, hue=piece_stats.index, palette='Reds_r', legend=False)
        plt.xticks(rotation=45, ha='right')
        plt.title(f"Piece Performance in {phase} (Avg Eval Change)")
        plt.xlabel("Piece")
        plt.ylabel("Avg Eval Change")
        plt.tight_layout()
        plt.show()

def plot_time_vs_quality(df_moves):
    """Correlates time spent on move vs how good the move was."""
    player_moves = df_moves[(df_moves['players_move'] == True) & (df_moves['time_used_sec'].notnull())]
    
    # Filter out extreme outliers for visualization
    plot_data = player_moves[player_moves['time_used_sec'] < 120] # Moves under 2 mins
    
    corr, _ = pearsonr(player_moves['time_used_sec'], player_moves['move_eval_change'])
    print(f"Correlation between Time Used and Move Quality (Eval Change): {corr:.4f}")
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=plot_data, x='time_used_sec', y='move_eval_change', alpha=0.3)
    plt.title(f"Time Used vs Move Quality (Correlation: {corr:.2f})")
    plt.xlabel("Time Used (sec)")
    plt.ylabel("Eval Change (Centipawns)")
    plt.ylim(-1000, 500) # Limit y-axis to focus on common mistakes vs good moves
    plt.show()

def plot_hourly_winrate(df_games):
    """Relation between hour of the day and win chance."""
    hourly_stats = df_games.groupby('day_hour')['is_win'].mean() * 100
    
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=hourly_stats.index, y=hourly_stats.values, marker='o', linewidth=2)
    plt.title("Win Rate by Hour of Day")
    plt.xlabel("Hour (0-23)")
    plt.ylabel("Win Rate (%)")
    plt.xticks(range(0, 24))
    plt.grid(True)
    plt.show()

def analyze_castling(df_games):
    """Analyzes win rates based on castling choices."""
    # Create categories
    conditions = [
        (df_games['player_short_castle']),
        (df_games['player_long_castle']),
        (~df_games['player_short_castle'] & ~df_games['player_long_castle'])
    ]
    choices = ['Short', 'Long', 'None']
    df_games['my_castle'] = np.select(conditions, choices, default='Unknown')
    
    # Same side vs Opposite
    def get_opposition(row):
        if row['my_castle'] == 'None': return 'No Castle'
        
        opp_castle = 'None'
        if row['opponent_short_castle']: opp_castle = 'Short'
        elif row['opponent_long_castle']: opp_castle = 'Long'
        
        if opp_castle == 'None': return 'Opponent No Castle'
        if row['my_castle'] == opp_castle: return 'Same Side'
        return 'Opposite Side'

    df_games['castle_scenario'] = df_games.apply(get_opposition, axis=1)
    
    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Win rate by my castle type
    castle_stats = df_games.groupby('my_castle')['is_win'].mean() * 100
    sns.barplot(x=castle_stats.index, y=castle_stats.values, ax=axes[0], hue=castle_stats.index, palette='Blues', legend=False)
    axes[0].set_title("Win Rate by My Castle Type")
    axes[0].set_ylabel("Win Rate (%)")
    axes[0].tick_params(axis='x', rotation=45)
    
    # Win rate by scenario
    scenario_stats = df_games.groupby('castle_scenario')['is_win'].mean() * 100
    sns.barplot(x=scenario_stats.index, y=scenario_stats.values, ax=axes[1], hue=scenario_stats.index, palette='Purples', legend=False)
    axes[1].set_title("Win Rate by Castling Scenario")
    axes[1].set_ylabel("Win Rate (%)")
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def analyze_blunders(df_moves):
    """
    Identifies which pieces are most likely to be blundered.
    Blunder defined as eval drop < -200 (approx 2 pawns).
    """
    player_moves = df_moves[df_moves['players_move'] == True]
    
    # Define blunder
    BLUNDER_THRESHOLD = -200
    blunders = player_moves[player_moves['move_eval_change'] < BLUNDER_THRESHOLD]
    
    # 1. Likelihood: Blunders per move for each piece type
    piece_move_counts = player_moves['moved_piece'].value_counts()
    piece_blunder_counts = blunders['moved_piece'].value_counts()
    
    blunder_rate = (piece_blunder_counts / piece_move_counts).sort_values(ascending=False)
    
    # 2. Severity: Average eval loss when blundering with that piece
    blunder_severity = blunders.groupby('moved_piece')['move_eval_change'].mean().sort_values()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.barplot(x=blunder_rate.index, y=blunder_rate.values, ax=axes[0], hue=blunder_rate.index, palette='Reds', legend=False)
    axes[0].set_title("Blunder Likelihood (Blunders / Total Moves)")
    axes[0].set_ylabel("Frequency")
    axes[0].tick_params(axis='x', rotation=45)
    
    sns.barplot(x=blunder_severity.index, y=blunder_severity.values, ax=axes[1], hue=blunder_severity.index, palette='Reds_r', legend=False)
    axes[1].set_title("Blunder Severity (Avg Eval Loss)")
    axes[1].set_ylabel("Avg Centipawn Loss")
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def analyze_missed_mates(df_moves):
    """
    Analyzes how many mates in one were missed.
    Condition: Opponent's move left a 'Mate in 1' for me, but I didn't play 'best_move' 
    (assuming best move was the mate).
    """
    # Sort by game and move order to ensure correct sequence
    df_sorted = df_moves.sort_values(['game_url', 'move_order'])
    
    missed_mates_count = 0
    opportunities_count = 0
    
    # We need to look at pairs of moves: Opponent (Prev) -> Player (Curr)
    # Shifted DataFrame to align Previous move with Current move
    df_sorted['prev_mate_possible'] = df_sorted['mate_possible'].shift(1)
    df_sorted['prev_mate_in_moves'] = df_sorted['mate_in_moves'].shift(1)
    df_sorted['prev_pos_eval'] = df_sorted['pos_eval'].shift(1)
    df_sorted['prev_game_url'] = df_sorted['game_url'].shift(1)
    df_sorted['prev_players_move'] = df_sorted['players_move'].shift(1)
    
    # Filter for:
    # 1. Same game
    # 2. Current move is Player's move
    # 3. Previous move was Opponent's move (implied by 2 usually, but checked)
    # 4. Previous state (after opponent move) was Mate in 1 for Player. 
    #    - mate_possible == True
    #    - mate_in_moves == 1
    #    - pos_eval > 0 (Positive eval means advantage/mate for Player)
    
    candidates = df_sorted[
        (df_sorted['game_url'] == df_sorted['prev_game_url']) &
        (df_sorted['players_move'] == True) &
        (df_sorted['prev_mate_possible'] == True) &
        (df_sorted['prev_mate_in_moves'] == 1) &
        (df_sorted['prev_pos_eval'] > 0)
    ]
    
    opportunities_count = len(candidates)
    
    # Missed if I did NOT play the best move (assuming engine always finds the mate in 1 as best move)
    missed = candidates[candidates['is_best_move'] == False]
    missed_mates_count = len(missed)
    
    print(f"Mate in 1 Opportunities: {opportunities_count}")
    print(f"Mate in 1 Missed: {missed_mates_count}")
    
    if opportunities_count > 0:
        found_rate = ((opportunities_count - missed_mates_count) / opportunities_count) * 100
        print(f"Mate in 1 Spotting Rate: {found_rate:.1f}%")
        
        plt.figure(figsize=(6, 6))
        plt.pie([missed_mates_count, opportunities_count - missed_mates_count], 
                labels=['Missed', 'Found'], autopct='%1.1f%%', colors=['red', 'green'])
        plt.title("Mate in 1 Detection")
        plt.show()
    else:
        print("No Mate in 1 opportunities found in the dataset.")

def plot_opening_winrate(df_games, min_games=5):
    """Analyzes win rate by Opening (ECO)."""
    # Extract just the category (e.g., C44 -> C) or use full ECO? 
    # Using full ECO might be too granular. Let's try full ECO first but filter.
    
    opening_stats = df_games.groupby('eco').agg(
        games=('game_url', 'count'),
        wins=('is_win', 'sum')
    )
    opening_stats['win_rate'] = (opening_stats['wins'] / opening_stats['games']) * 100
    
    # Filter
    top_openings = opening_stats[opening_stats['games'] >= min_games].sort_values('win_rate', ascending=False)
    
    if top_openings.empty:
        print(f"No openings with at least {min_games} games.")
        return

    # Take top 10 and bottom 10 if too many
    if len(top_openings) > 20:
        subset = pd.concat([top_openings.head(10), top_openings.tail(10)])
    else:
        subset = top_openings

    plt.figure(figsize=(12, 6))
    sns.barplot(x=subset.index, y=subset['win_rate'], hue=subset.index, palette="Spectral", legend=False)
    plt.xticks(rotation=90)
    plt.title(f"Win Rate by Opening ECO (min {min_games} games)")
    plt.ylabel("Win Rate (%)")
    plt.xlabel("ECO Code")
    plt.tight_layout()
    plt.show()

def analyze_ragequits(df_games, df_moves, eval_threshold=500, time_threshold=10):
    """
    Analyzes 'ragequits'.
    Definitions:
    1. Abandoned: Opponent result is 'abandoned' AND last evaluation is > eval_threshold (in player's favor).
    2. Stall/Timeout: Opponent result is 'timeout' AND last evaluation is > eval_threshold AND 
       opponent had > time_threshold seconds on their previous move.
    """
    print(f"Analyzing Ragequits (Eval Threshold: {eval_threshold}, Time Threshold: {time_threshold}s)...")
    
    # Get last move of each game to check evaluation
    last_moves = df_moves.sort_values('move_order').groupby('game_url').last()[['pos_eval', 'players_move']]
    
    # Get last move made by OPPONENT to check their clock (for timeout analysis)
    # We want the opponent's clock time at the end of their last move.
    opp_moves = df_moves[df_moves['players_move'] == False]
    last_opp_moves = opp_moves.sort_values('move_order').groupby('game_url').last()[['time_on_clock_sec']]
    last_opp_moves.rename(columns={'time_on_clock_sec': 'opp_last_clock'}, inplace=True)
    
    # Merge data
    df_merged = df_games.merge(last_moves, on='game_url', how='left')
    df_merged = df_merged.merge(last_opp_moves, on='game_url', how='left')
    
    # 1. Abandoned Ragequits
    # Result is abandoned, and position is winning for player
    abandoned_rq = df_merged[
        (df_merged['opponent_result'] == 'abandoned') & 
        (df_merged['pos_eval'] > eval_threshold)
    ]
    
    # 2. Timeout Ragequits (Stalling)
    # Result is timeout, position winning, it was opponent's turn (last move by player), and they had time
    timeout_rq = df_merged[
        (df_merged['opponent_result'] == 'timeout') & 
        (df_merged['pos_eval'] > eval_threshold) &
        (df_merged['players_move'] == True) & # Last recorded move was player's, so it was opponent's turn
        (df_merged['opp_last_clock'] > time_threshold)
    ]
    
    total_games = len(df_games)
    count_abandoned = len(abandoned_rq)
    count_timeout = len(timeout_rq)
    total_rq = count_abandoned + count_timeout
    rq_rate = (total_rq / total_games) * 100
    
    print(f"Total Games: {total_games}")
    print(f"Ragequits detected: {total_rq} ({rq_rate:.2f}%)")
    print(f"  - Abandoned in lost position: {count_abandoned}")
    print(f"  - Stalled/Timeout in lost position (>10s left): {count_timeout}")
    
    # Visualization
    categories = ['Normal Games', 'Ragequits']
    values = [total_games - total_rq, total_rq]
    
    plt.figure(figsize=(8, 5))
    plt.pie(values, labels=categories, autopct='%1.1f%%', colors=['lightgray', 'crimson'], startangle=140)
    plt.title("Ragequit Percentage")
    plt.show()
    
    if total_rq > 0:
        rq_types = ['Abandoned', 'Timeout (Stall)']
        rq_counts = [count_abandoned, count_timeout]
        
        plt.figure(figsize=(8, 5))
        sns.barplot(x=rq_types, y=rq_counts, hue=rq_types, palette='Reds_r', legend=False)
        plt.title("Ragequit Types")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()
