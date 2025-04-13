import chess
import chess.svg
from pathlib import Path
import numpy as np

if __name__ == "__main__":

    n_folder = Path(".") / "data" / "test_trajectories"
    a_folder = Path(".") / "data" / "altered_trajectories"

    n = np.load(n_folder / "trajectory_25.npy")[:1_000]
    a = np.load(a_folder / "trajectory_25.npy")

    def get_svg_sequence(trajectory, board_size=150):
        boards_svg = []
        board = chess.Board(trajectory[0])
        boards_svg.append(chess.svg.board(board=board, size=board_size))
        for move_uci in trajectory[1:]:

            if move_uci == "":
                break
            move = chess.Move.from_uci(move_uci)
            board.push(move)
            boards_svg.append(chess.svg.board(board=board, size=board_size, lastmove=move))
        return boards_svg

    html_parts = [
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>Altered Chess Trajectories</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; background-color: #f4f4f4; margin: 20px; }",
        "/* The scroll-container holds all the game boxes horizontally */",
        ".scroll-container { display: flex; overflow-x: auto; }",
        "/* Each game-box is a column showing a pair of trajectories side by side */",
        ".game-box { display: flex; flex-direction: row; border: 1px solid #ddd; border-radius: 8px; margin: 10px; padding: 10px; }",
        "/* Each trajectory is a vertical column of boards */",
        ".trajectory { display: flex; flex-direction: column; margin: 5px; }",
        ".trajectory svg { margin-bottom: 10px; }",
        "</style>",
        "</head>",
        "<body>",
        "<h2>Altered Chess Trajectories</h2>",
        "<div class='scroll-container'>"
    ]


    altered_mask = ((n[:, 1:] != a[:, 1:]).any(axis=1))
    si = np.random.choice(np.where(altered_mask)[0], size=5, replace=False)

    for i in si:
        traj_normal = n[i]
        traj_altered = a[i]

        svg_normal = get_svg_sequence(traj_normal)
        svg_altered = get_svg_sequence(traj_altered)

        game_html = f"<h3>{i}</h3><div class='game-box'>"

        game_html += "<div class='trajectory'>"
        for svg in svg_normal:
            game_html += svg
        game_html += "</div>"

        game_html += "<div class='trajectory'>"
        for svg in svg_altered:
            game_html += svg
        game_html += "</div>"

        game_html += "</div>"
        html_parts.append(game_html)

    html_parts.extend([
        "</div>",
        "</body>",
        "</html>"
    ])

    html_content = "\n".join(html_parts)
    output_file = Path("altered_chess_trajectories.html")
    output_file.write_text(html_content, encoding="utf-8")
