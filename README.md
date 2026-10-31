# Chess Game Analysis Tool

A comprehensive tool for fetching, analyzing, and visualizing chess games from Chess.com using Stockfish and Python's data science stack.

## Overview

This project provides a full pipeline to:
1. **Fetch:** Download chess games from a specific Chess.com user.
2. **Analyze:** Evaluate each move in the downloaded games using the Stockfish engine.
3. **Visualize:** Generate deep insights into playing patterns, blunder tendencies, time management, and more through various statistical plots and hypothesis tests.

## Features

- **Automated Data Retrieval:** Fetches PGN data directly from Chess.com archives.
- **Deep Engine Analysis:** Uses Stockfish to calculate centipawn loss, identify best moves, and detect blunders.
- **Multi-processing Support:** Parallelizes game analysis to leverage multiple CPU cores.
- **Advanced Visualizations:**
  - Overall win rates and performance by opponent country.
  - Performance by game phase (Opening, Middlegame, Endgame).
  - Time usage vs. move quality correlation.
  - Blunder analysis by piece type and severity.
  - Castling strategy effectiveness.
  - Opening (ECO) performance and frequency.
- **Statistical Testing:**
  - Color advantage (White vs. Black) significance.
  - Time pressure impact on move quality.
  - Logistic regression for win prediction based on rating and color.

## Project Structure

- `main.py`: The entry point that orchestrates fetching and analyzing games.
- `game_load.py`: Handles fetching games from Chess.com API.
- `stockfish_eval.py`: Manages the Stockfish engine interface and game evaluation logic.
- `data_analysis.py`: Contains extensive logic for data loading, visualization, and statistical modeling.
- `main_notebook.ipynb`: A Jupyter notebook for interactive exploration and visualization of results.
- `stockfish/`: Directory containing the Stockfish engine executable.
- `requirements.txt`: Python dependencies.

## Prerequisites

- Python 3.8+
- Stockfish engine (Windows executable included in `stockfish/` directory).

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd DMB2_semestralni_prace
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Create a `.env` file in the root directory to configure the tool. Below are the variables used in the code:

```env
# Mandatory: The Chess.com username whose games will be analyzed.
# If not set, the program will terminate.
CHESS_USER=your_chess_username

# Optional: Directory for raw games. 
# Defaults to "{CHESS_USER}_data/games" if not provided or if the path doesn't exist.
GAMES_PATH=path/to/save/raw/games

# Optional: Directory for analyzed games. 
# Defaults to "{CHESS_USER}_data/games_analyzed" if not provided or if the path doesn't exist.
ANALYZED_PATH=path/to/save/analyzed/games

# Optional: The search depth for the Stockfish engine. 
# Used in stockfish_eval.py for transform_eval (max_value calculation) and Stockfish initialization.
# Defaults to 10 in some contexts, but initialized with 15 in analyze_game.
STOCKFISH_DEPTH=15

# Optional: Number of threads for the Stockfish engine.
# Defaults to 1 if not set.
STOCKFISH_THREADS=4

# Optional: Number of parallel processes for analyzing archives.
# Defaults to 1 if not set. Total CPU usage = ANALYSIS_PROCESSES_COUNT * STOCKFISH_THREADS.
ANALYSIS_PROCESSES_COUNT=4

# Optional: Number of parallel processes for fetching games from Chess.com.
# Defaults to 1 if not set.
FETCH_PROCESSES_COUNT=4

# Optional: Time in milliseconds to wait between retries when hitting Chess.com rate limits (429).
# Defaults to 1000ms if not set.
FETCH_RETRY_WAIT_TIME=1000

# Optional: Time control filter for fetching (e.g., "180" for 3 min blitz).
# Also used as a fallback for move time calculations if %clk is missing.
# Defaults to 180 if not set.
FETCH_TIME_CONTROL=180
```

## Usage

### 1. Fetch and Analyze Games
Run the main script to download and analyze games:
```bash
python main.py
```

### 2. Run Data Analysis
You can use the provided Jupyter notebook or run specific analysis functions in a script:
```bash
# Open the notebook
jupyter notebook main_notebook.ipynb
```

## Analysis Examples

The `data_analysis.py` module includes functions for:
- `plot_winrate_overall(df_games)`
- `analyze_game_phases(df_moves)`
- `plot_time_vs_quality(df_moves)`
- `analyze_blunders(df_moves)`
- `test_color_advantage(df_games)`

## License

This project is for educational purposes as part of the DMB2 course.
