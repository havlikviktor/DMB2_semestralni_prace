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

Create a `.env` file in the root directory with the following variables:

```env
CHESS_USER=your_chess_username
STOCKFISH_DEPTH=15
STOCKFISH_THREADS=4
ANALYSIS_PROCESSES_COUNT=4
FETCH_TIME_CONTROL=600
GAMES_PATH=path/to/save/raw/games
ANALYZED_PATH=path/to/save/analyzed/games
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
