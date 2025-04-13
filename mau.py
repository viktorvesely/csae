import pandas as pd
import numpy as np
from pathlib import Path
import chess
import chess.svg
import chess.engine
from scipy.sparse import csr_matrix

def load_data():
    df = pd.read_parquet(Path(".") / "data" / "mau.parquet")
    return df

def build_csr_matrix(df: pd.DataFrame) -> csr_matrix:

    mau_cols = [f"mau{i}" for i in range(1, 11)]
    act_cols = [f"a{i}" for i in range(1, 11)]

    max_index = df[mau_cols].values.max()
    n_rows = len(df)

    row_indices = []
    col_indices = []
    data_values = []

    for mau_col, act_col in zip(mau_cols, act_cols):
        rows = np.arange(n_rows)
        cols = df[mau_col].values
        vals = df[act_col].values.astype(np.float32)

        row_indices.append(rows)
        col_indices.append(cols)
        data_values.append(vals)

    row_indices = np.concatenate(row_indices)
    col_indices = np.concatenate(col_indices)
    data_values = np.concatenate(data_values)


    latent_matrix = csr_matrix(
        (data_values, (row_indices, col_indices)),
        shape=(n_rows, max_index + 1)
    )

    return latent_matrix

def bsv(indices: list[int], dim: int) -> csr_matrix:
    # manually buiodl sparse vector
    n = len(indices)
    row = np.zeros(n, dtype=int)
    values = np.full(n, 0.8)
    return csr_matrix((values, (row, indices)), shape=(1, dim))

def find_k_nearest_neighbors(
        latent_matrix: csr_matrix,
        row_id: int = None,
        k: int = 30,
        query_vector: csr_matrix = None
    ):
    row_norms = np.ravel(latent_matrix.multiply(latent_matrix).sum(axis=1))

    if query_vector is not None:
        q_norm = query_vector.multiply(query_vector).sum()
        dot_products = query_vector.dot(latent_matrix.T).toarray().ravel()
    else:
        query_vector = latent_matrix.getrow(row_id)
        q_norm = row_norms[row_id]
        dot_products = query_vector.dot(latent_matrix.T).toarray().ravel()

    dist_sq = q_norm + row_norms - 2.0 * dot_products
    nn_indices = np.argsort(dist_sq)[:k]
    return nn_indices

def evaluate_board(board: chess.Board, engine: chess.engine.SimpleEngine) -> str:

    info = engine.analyse(board, chess.engine.Limit(time=0.1))
    score = info["score"].white()
    if score.is_mate():
        return f"M{score.mate()}"
    else:
        return f"{score.score() / 100:.2f}"

def main():
    df = load_data()

    latent_matrix = build_csr_matrix(df)

    if True:
        dim = latent_matrix.shape[1]
        query_vector = bsv([91372, 58797, 116521, 72686, 120044, 14926, 105060, 86373, 105059, 97473], dim)
        neighbor_indices = find_k_nearest_neighbors(latent_matrix, query_vector=query_vector, k=10)
    else:
        random_row = np.random.choice(df.shape[0])
        neighbor_indices = find_k_nearest_neighbors(latent_matrix, random_row, k=10)

    # Find most overlapping MAUs
    cluster_df = df.iloc[neighbor_indices].copy()
    mau_cols = [f"mau{i}" for i in range(1, 11)]
    top_units_series = pd.Series(cluster_df[mau_cols].values.flatten())
    top_units_counts = top_units_series.value_counts().head(10)
    overlaps_str = ", ".join([f"{unit}({cnt})" for unit, cnt in top_units_counts.items()])

    # Works only on windows
    engine = chess.engine.SimpleEngine.popen_uci("./stock.exe")

    html_parts = [
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>Chess Boards</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; background-color: #f4f4f4; margin: 20px; }",
        ".header { text-align: center; margin-bottom: 30px; }",
        ".board-grid { display: flex; flex-wrap: wrap; justify-content: center; }",
        ".board-container { background: #fff; border: 1px solid #ddd; border-radius: 8px; "
        "box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 10px; padding: 15px; width: 460px; box-sizing: border-box; }",
        ".board-container svg { display: block; }",
        ".boards { display: flex; justify-content: space-between; }",
        ".eval { text-align: center; margin-top: 10px; font-size: 1.1em; font-weight: bold; color: #333; }",
        "</style>",
        "</head>",
        "<body>",
        f"<div class='header'><h2>Top Overlapping Units: {overlaps_str}</h2></div>",
        "<div class='board-grid'>"
    ]


    for idx in cluster_df.index:
        fen_str = df.at[idx, "fens"]
        move_uci = df.at[idx, "move"]

        board_before = chess.Board(fen_str)
        svg_before = chess.svg.board(board=board_before, size=200)
        evaluation = evaluate_board(board_before, engine)

        if pd.notna(move_uci) and move_uci:
            move = chess.Move.from_uci(move_uci)
            board_after = chess.Board(fen_str)
            board_after.push(move)
            svg_after = chess.svg.board(board=board_after, size=200, lastmove=move)
            e_after = evaluate_board(board_after, engine)
        else:
            svg_after = ""

        board_html = (
            "<div class='board-container'>"
            "<div class='boards'>"
            f"<div>{svg_before}</div>"
            f"<div>{svg_after}</div>"
            "</div>"
            f"<div class='eval'>{evaluation} || {e_after} </div>"
            "</div>"
        )
        html_parts.append(board_html)

    html_parts.extend([
        "</div>",
        "</body>",
        "</html>"
    ])

    html_content = "\n".join(html_parts)
    output_file = Path("output.html")
    output_file.write_text(html_content, encoding="utf-8")

    print("HTML file written to", output_file.resolve())
    engine.quit()

if __name__ == '__main__':
    main()
