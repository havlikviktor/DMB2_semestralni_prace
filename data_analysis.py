import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from scipy.stats import pearsonr, ttest_ind, chi2_contingency, f_oneway

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
        plt.text(i, v + 0.3, f"{v:.1f}%", ha='center', va='bottom', fontweight='bold')
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
    # Convert to percentage
    phase_stats_accuracy['accuracy_pct'] = phase_stats_accuracy['is_best_move'] * 100
    
    sns.barplot(data=phase_stats_accuracy, x='phase', y='accuracy_pct', ax=axes[1], order=['Opening', 'Middlegame', 'Endgame'], hue='phase', palette='viridis', legend=False)
    axes[1].set_title("Move Accuracy (% Best Move) by Phase")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 100)
    
    # Use the actual plotted values for labels to ensure match
    for i, row in phase_stats_accuracy.iterrows():
        # Match index to order
        phase = row['phase']
        if phase == 'Opening': x_pos = 0
        elif phase == 'Middlegame': x_pos = 1
        else: x_pos = 2
        
        val = row['accuracy_pct']
        axes[1].text(x_pos, val + 1, f"{val:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.show()
    
    phases = ['Opening', 'Middlegame', 'Endgame']
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    for i, phase in enumerate(phases):
        ax = axes[i]
        phase_data = player_moves[player_moves['phase'] == phase]
        
        if phase_data.empty:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            ax.set_title(f"{phase} (No Data)")
            continue
            
        piece_stats = phase_data.groupby('moved_piece')['move_eval_change'].mean().sort_values()
        
        # Plot for each phase on the specific subplot axis
        sns.barplot(x=piece_stats.index, y=piece_stats.values, ax=ax, hue=piece_stats.index, palette='Reds_r', legend=False)
        ax.set_title(f"Piece Performance in {phase} (Avg Eval Change)")
        ax.set_xlabel("Piece")
        ax.set_ylabel("Avg Eval Change")
        ax.tick_params(axis='x', rotation=45)
        
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

def analyze_blunders(df_moves, blunder_threshold=-200):
    """
    Identifies which pieces are most likely to be blundered.
    Blunder defined as eval drop < -200 (approx 2 pawns).
    Separates blunders into two categories: those involving mate scores and normal eval drops.
    """
    player_moves = df_moves[df_moves['players_move'] == True]
    blunders = player_moves[player_moves['move_eval_change'] < blunder_threshold].copy()
    
    if blunders.empty:
        print("No blunders found matching the threshold.")
        return

    # Determine if a blunder is mate-related.
    # It is mate-related if:
    # 1. The resulting position allows a mate (mate_possible is True).
    # 2. Or if the evaluation swing is massive (implies previous position was mate or near-mate).
    # Using 2000 as a cutoff (20 pawns) to distinguish "mate logic" scores from "positional" scores.
    blunders['is_mate_involved'] = blunders['mate_possible'] | (blunders['move_eval_change'].abs() > 2000)
    
    blunders_no_mate = blunders[~blunders['is_mate_involved']]
    blunders_mate = blunders[blunders['is_mate_involved']]

    # 1. Likelihood: Blunders per move for each piece type (Overall)
    piece_move_counts = player_moves['moved_piece'].value_counts()
    piece_blunder_counts = blunders['moved_piece'].value_counts()
    
    blunder_rate = (piece_blunder_counts / piece_move_counts).sort_values(ascending=False)
    
    # 2. Severity: Average eval loss when blundering with that piece
    severity_no_mate = blunders_no_mate.groupby('moved_piece')['move_eval_change'].mean().sort_values()
    severity_mate = blunders_mate.groupby('moved_piece')['move_eval_change'].mean().sort_values()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Frequency
    sns.barplot(x=blunder_rate.index, y=blunder_rate.values, ax=axes[0], hue=blunder_rate.index, palette='Reds', legend=False)
    axes[0].set_title("Blunder Likelihood (Overall)")
    axes[0].set_ylabel("Frequency (Blunders / Moves)")
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Severity (No Mate)
    if not severity_no_mate.empty:
        sns.barplot(x=severity_no_mate.index, y=severity_no_mate.values, ax=axes[1], hue=severity_no_mate.index, palette='Reds_r', legend=False)
        axes[1].set_title("Blunder Severity (No Mate Involved)")
        axes[1].set_ylabel("Avg Centipawn Loss")
        axes[1].tick_params(axis='x', rotation=45)
    else:
        axes[1].set_title("No non-mate blunders")

    # Plot 3: Severity (Mate)
    if not severity_mate.empty:
        sns.barplot(x=severity_mate.index, y=severity_mate.values, ax=axes[2], hue=severity_mate.index, palette='Dark2_r', legend=False)
        axes[2].set_title("Blunder Severity (Mate Involved)")
        axes[2].set_ylabel("Avg Mate Distance (M)")
        axes[2].tick_params(axis='x', rotation=45)
        
        # MATE_MAX assumes standard config from stockfish_eval.py (Depth 10 * 1000 = 10000)
        MATE_MAX = int(os.getenv("STOCKFISH_DEPTH", 10)) * 1000 
        
        def mate_formatter(x, pos):
            if x == 0: return "0"
            val = abs(x)
            # If value is very low (e.g. < 500), it's likely M20+ or noise, showing raw might be better
            # But user asked for transformation. M20 is 500.
            if val < 100: return f"{int(x)}"
            
            mate_in_x = MATE_MAX / val
            # Since values are negative for blunders (axis goes down), we label it M...
            # We can prefix with sign if we want, but "M2" usually implies the magnitude.
            return f"M{mate_in_x:.1f}"

        axes[2].yaxis.set_major_formatter(FuncFormatter(mate_formatter))

    else:
        axes[2].set_title("No mate blunders")
    
    plt.tight_layout()
    plt.show()

def analyze_missed_mates(df_moves):
    """
    Analyzes missed mate sequences.
    Identifies positions where a forced mate was available (Mate in 1, 2, 3+)
    and checks if the player failed to play the best move (missing the optimal mate).
    """
    # Sort by game and move order
    df_sorted = df_moves.sort_values(['game_url', 'move_order'])
    
    # We need to look at the state BEFORE the player's move.
    # The 'pos_eval' and 'mate_in_moves' in our current row reflect the state AFTER the move.
    # So we look at the PREVIOUS row (opponent's move) to see what the board looked like for the player.
    
    df_sorted['prev_mate_possible'] = df_sorted['mate_possible'].shift(1)
    df_sorted['prev_mate_in_moves'] = df_sorted['mate_in_moves'].shift(1)
    df_sorted['prev_pos_eval'] = df_sorted['pos_eval'].shift(1)
    df_sorted['prev_game_url'] = df_sorted['game_url'].shift(1)
    
    # Filter for Player's moves where a mate was possible in the previous state
    # Positive eval for player means the mate is FOR the player.
    # Note: Stockfish eval is usually relative to the side to move. 
    # If prev move was opponent's, and eval is positive, it might mean opponent is winning? 
    # Let's rely on 'pos_eval' from our data processing which standardizes eval for the analyzer user (as_white).
    # If 'pos_eval' is positive, it's good for the player.
    
    mate_opportunities = df_sorted[
        (df_sorted['game_url'] == df_sorted['prev_game_url']) &
        (df_sorted['players_move'] == True) &
        (df_sorted['prev_mate_possible'] == True) &
        (df_sorted['prev_pos_eval'] > 0) # Mate is for the player
    ].copy()
    
    if mate_opportunities.empty:
        print("No forced mate opportunities found.")
        return

    # Categorize by Mate Distance
    def get_mate_category(moves):
        if moves == 1: return "Mate in 1"
        if moves == 2: return "Mate in 2"
        if moves == 3: return "Mate in 3"
        return "Mate in 4+"

    mate_opportunities['mate_type'] = mate_opportunities['prev_mate_in_moves'].apply(get_mate_category)
    
    # 1. Optimal Find: Player played the best move
    mate_opportunities['found_optimal'] = mate_opportunities['is_best_move']
    
    # 2. Pattern Find: Player played a move that kept the forced mate (even if slower)
    # Check if the resulting position (current row) is still a mate for the player
    mate_opportunities['found_pattern'] = (
        (mate_opportunities['mate_possible'] == True) & 
        (mate_opportunities['pos_eval'] > 0)
    )
    # Note: If M1 was available, and you played a move that results in Mate, it's a win (game over).
    # In 'is_game_over' case, pos_eval might be max, but mate_possible might be False/True depending on implementation.
    # Let's check 'is_game_over' and result as well for M1 cases.
    mate_opportunities.loc[mate_opportunities['is_game_over'] & (mate_opportunities['pos_eval'] > 0), 'found_pattern'] = True

    # Aggregation
    stats = mate_opportunities.groupby('mate_type').agg(
        total=('game_url', 'count'),
        found_optimal=('found_optimal', 'sum'),
        found_pattern=('found_pattern', 'sum')
    )
    
    stats['optimal_rate'] = (stats['found_optimal'] / stats['total']) * 100
    stats['pattern_rate'] = (stats['found_pattern'] / stats['total']) * 100
    
    print("\n--- Missed Mate Analysis ---")
    print(stats[['total', 'found_optimal', 'found_pattern', 'optimal_rate', 'pattern_rate']])
    
    # Plotting
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 12))
    
    # 1. Optimal Rate %
    sns.barplot(x=stats.index, y=stats['optimal_rate'], ax=axes[0], hue=stats.index, palette="viridis", legend=False)
    axes[0].set_title("Optimal Mate Found Rate (%)")
    axes[0].set_ylabel("Found Rate (%)")
    axes[0].set_ylim(0, 100)
    for i, v in enumerate(stats['optimal_rate']):
        axes[0].text(i, v + 1, f"{v:.1f}%", ha='center')
        
    # 2. Pattern Rate %
    sns.barplot(x=stats.index, y=stats['pattern_rate'], ax=axes[1], hue=stats.index, palette="Blues", legend=False)
    axes[1].set_title("Mate Pattern Maintained Rate (%)")
    axes[1].set_ylabel("Maintained Rate (%)")
    axes[1].set_ylim(0, 100)
    for i, v in enumerate(stats['pattern_rate']):
        axes[1].text(i, v + 1, f"{v:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.show()

def analyze_strong_side(df_games, df_moves):
    """
    Comprehensive comparison between Player and Opponents.
    1. ACPL (Average Centipawn Loss) Comparison.
    2. Punishment Rate (Conversion of blunders).
    3. Win Rate by Rating Difference.
    """
    print("\n--- Strong Side Analysis (Player vs Opponent) ---")
    
    # --- 1. ACPL Calculation ---
    # Prepare data
    # Player Loss: -move_eval_change (if change < 0)
    # Opponent Loss: +move_eval_change (if change > 0)
    # Clip at 1000 to avoid mate scores skewing averages
    
    df_moves['cp_loss'] = 0.0
    
    # Player moves: Loss is negative change
    df_moves.loc[df_moves['players_move'] == True, 'cp_loss'] = \
        -df_moves.loc[df_moves['players_move'] == True, 'move_eval_change']
        
    # Opponent moves: Loss is positive change (eval goes up for player)
    df_moves.loc[df_moves['players_move'] == False, 'cp_loss'] = \
        df_moves.loc[df_moves['players_move'] == False, 'move_eval_change']
        
    # Filter only positive losses (mistakes) and clip
    df_moves['cp_loss'] = df_moves['cp_loss'].clip(lower=0, upper=1000)
    
    # Group by game and player
    acpl_stats = df_moves.groupby(['game_url', 'players_move'])['cp_loss'].mean().unstack()
    acpl_stats.rename(columns={True: 'Player_ACPL', False: 'Opponent_ACPL'}, inplace=True)
    
    # Merge with game result
    acpl_stats = acpl_stats.merge(df_games[['game_url', 'is_win', 'player_result']], on='game_url')
    
    # Plot 1: ACPL Distribution
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=acpl_stats['Player_ACPL'], label='Me', fill=True, color='blue', alpha=0.3)
    sns.kdeplot(data=acpl_stats['Opponent_ACPL'], label='Opponent', fill=True, color='red', alpha=0.3)
    plt.title("Accuracy Comparison (Average Centipawn Loss)")
    plt.xlabel("ACPL (Lower is Better)")
    plt.xlim(0, 200) # Focus on normal range
    plt.legend()
    plt.show()
    
    print(f"My Avg ACPL: {acpl_stats['Player_ACPL'].mean():.2f}")
    print(f"Opponent Avg ACPL: {acpl_stats['Opponent_ACPL'].mean():.2f}")
    
    # --- 2. Punishment Rate ---
    # Did I win when opponent blundered? Did they win when I blundered?
    # Blunder defined as > 300 cp loss in a single move
    
    # Identify games with blunders
    blunder_games = df_moves[df_moves['cp_loss'] >= 300].groupby(['game_url', 'players_move']).size().unstack(fill_value=0)
    blunder_games.rename(columns={True: 'My_Blunders', False: 'Opp_Blunders'}, inplace=True)
    
    # Merge with results
    blunder_analysis = df_games[['game_url', 'is_win']].merge(blunder_games, on='game_url', how='left').fillna(0)
    
    # Scenarios
    # A: Opponent Blundered at least once. Did I win?
    opp_blundered = blunder_analysis[blunder_analysis['Opp_Blunders'] > 0]
    punish_rate = opp_blundered['is_win'].mean() * 100 if not opp_blundered.empty else 0
    
    # B: I Blundered at least once. Did I LOSE? (is_win == False)
    me_blundered = blunder_analysis[blunder_analysis['My_Blunders'] > 0]
    # Assuming is_win False = Loss or Draw. Strictly "Punished" means I didn't win.
    get_punished_rate = (1 - me_blundered['is_win'].mean()) * 100 if not me_blundered.empty else 0
    
    print(f"Punishment Rate (I win when they blunder): {punish_rate:.2f}%")
    print(f"Survival Rate (I win even if I blunder): {100-get_punished_rate:.2f}%")
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(['I Punish Them', 'They Punish Me'], [punish_rate, get_punished_rate], color=['green', 'orange'])
    plt.title("Blunder Punishment Rate")
    plt.ylabel("Win/Loss Rate (%)")
    plt.ylim(0, 100)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.1f}%', ha='center', va='bottom')
    plt.show()
    
    # --- 3. Rating Difference Performance ---
    df_games['rating_diff'] = df_games['player_rating'] - df_games['opponent_rating']
    
    # Bins
    bins = [-float('inf'), -200, -100, -50, 50, 100, 200, float('inf')]
    labels = ['<-200', '-200 to -100', '-100 to -50', 'Even (+/-50)', '+50 to +100', '+100 to +200', '>+200']
    
    df_games['rating_bucket'] = pd.cut(df_games['rating_diff'], bins=bins, labels=labels)
    
    rating_stats = df_games.groupby('rating_bucket', observed=False)['is_win'].agg(['mean', 'count'])
    rating_stats['win_rate'] = rating_stats['mean'] * 100
    
    # Filter buckets with enough data
    rating_stats = rating_stats[rating_stats['count'] > 5]
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=rating_stats.index, y=rating_stats['win_rate'], hue=rating_stats.index, palette='RdYlGn', legend=False)
    plt.title("Win Rate by Rating Difference")
    plt.xlabel("Rating Difference (Me - Opponent)")
    plt.ylabel("Win Rate (%)")
    plt.axhline(50, color='black', linestyle='--')
    plt.xticks(rotation=45)
    
    for i, v in enumerate(rating_stats['win_rate']):
        plt.text(i, v + 0.3, f"{v:.1f}%", ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.show()

def plot_opening_winrate(df_games, min_games=5):
    """Analyzes win rate by Opening (ECO) and shows game counts on a secondary axis."""
    
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

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot 1: Bar chart for Win Rate
    sns.barplot(x=subset.index, y=subset['win_rate'], ax=ax1, hue=subset.index, palette="Spectral", legend=False, alpha=0.9)
    ax1.set_xlabel("ECO Code")
    ax1.set_ylabel("Win Rate (%)", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_title(f"Win Rate by Opening ECO (min {min_games} games) with Counts")
    ax1.tick_params(axis='x')
    ax1.set_ylim(0, 115) # Extra space for labels

    # Plot 2: Line chart for Game Counts
    ax2 = ax1.twinx()
    sns.lineplot(
        x=range(len(subset)),
        y=subset['games'],
        ax=ax2,
        color='black',
        marker='o',
        linewidth=1
    )
    ax2.set_ylabel("Number of Games", color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    
    # Calculate offset for labels (5% of max games)
    y_offset = subset['games'].max() * 0.01

    # Add labels
    for i, (idx, row) in enumerate(subset.iterrows()):
        # Label on bar (Win Rate)
        ax1.text(i, row['win_rate'] + 2, f"{row['win_rate']:.1f}%", ha='center', va='bottom', color='black', fontsize=9)
        # Label on line (Count)
        ax2.text(i, row['games'] + y_offset, f"{row['games']}", ha='center', va='bottom', color='black', fontsize=9)

    plt.tight_layout()
    plt.show()

def plot_most_played_openings(df_games, top_n=15):
    """Plots the most frequently played openings (ECO) along with their win rates."""
    # Get top N openings by count
    opening_counts = df_games['eco'].value_counts().head(top_n)
    
    if opening_counts.empty:
        print("No opening data available.")
        return

    # Calculate win rate for these specific openings
    top_openings_df = df_games[df_games['eco'].isin(opening_counts.index)]
    
    stats = top_openings_df.groupby('eco').agg(
        games=('game_url', 'count'),
        wins=('is_win', 'sum')
    )
    stats['win_rate'] = (stats['wins'] / stats['games']) * 100
    
    # Reorder stats to match the order of opening_counts (descending by count)
    stats = stats.reindex(opening_counts.index)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot 1: Bar chart for Counts
    sns.barplot(x=stats.index, y=stats['games'], ax=ax1, hue=stats.index, palette="viridis", legend=False, alpha=0.8)
    ax1.set_xlabel("ECO Code")
    ax1.set_ylabel("Number of Games", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_title(f"Top {top_n} Most Played Openings (Count vs Win Rate)")
    ax1.tick_params(axis='x', rotation=45)

    # Plot 2: Line chart for Win Rate
    ax2 = ax1.twinx()
    # sns.lineplot might try to sort x-axis, so we pass range(len) as x and set labels manually or trust seaborn with categorical
    # Using pointplot or lineplot with categorical x usually works if data is set up right, but passing x=stats.index acts categorically in recent seaborn.
    # To be safe against seaborn sorting, we plot against range
    sns.lineplot(x=range(len(stats)), y=stats['win_rate'], ax=ax2, color='tab:red', marker='o', linewidth=2)
    ax2.set_ylabel("Win Rate (%)", color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim(0, 105) # Little bit over 100 for labels

    # Add count and winrate labels
    for i, (idx, row) in enumerate(stats.iterrows()):
        # Label on bar (Count)
        ax1.text(i, row['games'], f"{row['games']}", ha='center', va='bottom', color='black', fontsize=9)
        # Label on line (Win Rate)
        ax2.text(i, row['win_rate'] + 2, f"{row['win_rate']:.1f}%", ha='center', va='bottom', color='tab:red', fontsize=9, fontweight='bold')
        
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

def test_time_pressure_vs_quality(df_moves):
    """
    Test 1: Time Pressure vs Move Quality
    H0: Move quality (eval change) is the same for low time (<15s) and high time (>60s).
    H1: Move quality is different.
    """
    print("\n[Statistical hypothesis test] Time Pressure vs Move Quality")
    player_moves = df_moves[df_moves['players_move'] == True].copy()
    
    # Check available columns
    if 'time_on_clock_sec' in player_moves.columns:
        clock_col = 'time_on_clock_sec'
        threshold_low = 30
        threshold_high = 120
        print(f"Using Clock Time: Low < {threshold_low}s, High > {threshold_high}s")
    else:
        # Fallback to time used if clock not tracked properly (though it should be)
        clock_col = 'time_used_sec' 
        threshold_low = 5
        threshold_high = 60
        print(f"Using Move Time: Fast < {threshold_low}s, Slow > {threshold_high}s")

    group_pressure = player_moves[player_moves[clock_col] < threshold_low]['move_eval_change'].dropna()
    group_calm = player_moves[player_moves[clock_col] > threshold_high]['move_eval_change'].dropna()
    
    if len(group_pressure) > 10 and len(group_calm) > 10:
        stat, p_val = ttest_ind(group_pressure, group_calm, equal_var=False)
        print(f"Samples: Pressure (n={len(group_pressure)}), Calm (n={len(group_calm)})")
        print(f"Means: Pressure={group_pressure.mean():.2f}, Calm={group_calm.mean():.2f}")
        print(f"T-test result: statistic={stat:.4f}, p-value={p_val:.4e}")
        if p_val < 0.05:
            print(">> REJECT H0: Significant difference in move quality found.")
        else:
            print(">> FAIL TO REJECT H0: No significant difference found.")
    else:
        print("Not enough data for Time Pressure test.")

def test_color_advantage(df_games):
    """
    Test 2: Color Advantage (White vs Black Win Rate)
    H0: Win/Loss distribution is independent of color.
    H1: Win/Loss distribution depends on color.
    """
    print("\n[Statistical hypothesis test] White vs Black Advantage")
    
    # Contingency table: [ [White Wins, White Not-Wins], [Black Wins, Black Not-Wins] ]
    white_games = df_games[df_games['as_white'] == True]
    black_games = df_games[df_games['as_white'] == False]
    
    w_wins = white_games['is_win'].sum()
    w_total = len(white_games)
    b_wins = black_games['is_win'].sum()
    b_total = len(black_games)
    
    if w_total > 0 and b_total > 0:
        contingency = [
            [w_wins, w_total - w_wins],
            [b_wins, b_total - b_wins]
        ]
        stat, p_val, dof, expected = chi2_contingency(contingency)
        
        print(f"White Win Rate: {(w_wins/w_total)*100:.1f}% (n={w_total})")
        print(f"Black Win Rate: {(b_wins/b_total)*100:.1f}% (n={b_total})")
        print(f"Chi-Square result: statistic={stat:.4f}, p-value={p_val:.4e}")
        if p_val < 0.05:
            print(">> REJECT H0: Color significantly affects game result.")
        else:
            print(">> FAIL TO REJECT H0: No significant color advantage detected.")
    else:
        print("Not enough data for Color test.")

def test_phase_accuracy(df_moves):
    """
    Test 3: Phase Accuracy (ANOVA)
    H0: Mean move evaluation change is equal across Opening, Middlegame, Endgame.
    H1: At least one phase has a different mean accuracy.
    """
    print("\n[Statistical hypothesis test] Accuracy by Game Phase (ANOVA)")
    player_moves = df_moves[df_moves['players_move'] == True].copy()
    
    # Ensure phases are calculated
    if 'phase' not in player_moves.columns:
         player_moves['phase'] = player_moves.apply(classify_phase, axis=1)
    
    opening = player_moves[player_moves['phase'] == 'Opening']['move_eval_change'].dropna()
    midgame = player_moves[player_moves['phase'] == 'Middlegame']['move_eval_change'].dropna()
    endgame = player_moves[player_moves['phase'] == 'Endgame']['move_eval_change'].dropna()
    
    if len(opening) > 0 and len(midgame) > 0 and len(endgame) > 0:
        print(f"Means: Opening={opening.mean():.2f}, Middlegame={midgame.mean():.2f}, Endgame={endgame.mean():.2f}")
        stat, p_val = f_oneway(opening, midgame, endgame)
        print(f"ANOVA result: statistic={stat:.4f}, p-value={p_val:.4e}")
        if p_val < 0.05:
            print(">> REJECT H0: Significant difference in accuracy between phases.")
        else:
            print(">> FAIL TO REJECT H0: Accuracy is consistent across phases.")
    else:
        print("Not enough data for Phase ANOVA.")
