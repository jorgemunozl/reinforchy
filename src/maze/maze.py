import curses
import sys


MAZES = {
    "easy": """
####################
#@      #         G#
# ####### ##########
#                  #
# ###### ######### #
#      #         # #
###### # ####### # #
#      #       #   #
# ############ # ###
#                  #
####################
""".strip("\n"),
    "medium": """
####################
#@   #        #    #
### ### ####### ## #
#     #   #      # #
# ### ##### ###### #
# #       #      # #
# # ##### ###### # #
# #     #      # # #
# ##### ###### # # #
#     #      # #   #
##### ###### # #####
#         #  #     #
# ####### # ###### #
#       # #      # #
####### # ###### # #
#     # #      # # #
# ### # ###### # # #
# #   #      # #   #
# ########## # #####
#          G       #
####################
""".strip("\n"),
    "hard": """
############################
#@   #     #   #       #   #
### ### ### ### ##### ### ##
#   #   #   #   #   #     ##
## ##### ##### ### ####### #
#      #     #   #       # #
###### # ### ##### ##### # #
#    # # # #     # #   # # #
# ## # # # ##### # # # # # #
# ##   # #     # #   # #   #
# ###### ##### # ##### #####
#      #     # #     #     #
###### ##### # ##### ##### #
#    #     # #   #   #     #
# ## ##### # ### # ### ### #
# ##     # #   # #   # #   #
# ###### # ### # ### # # ###
#      # #   # #     # #   #
###### # ### # ####### ### #
#      #   # #       #   #G#
############################
""".strip("\n"),
}


def _parse_maze(s: str):
    grid = [list(row) for row in s.splitlines()]
    h = len(grid)
    w = len(grid[0]) if h else 0
    for row in grid:
        if len(row) != w:
            raise ValueError("Maze must be rectangular")

    start = None
    goal = None
    for y in range(h):
        for x in range(w):
            if grid[y][x] == "@":
                start = (y, x)
                grid[y][x] = " "
            elif grid[y][x] == "G":
                goal = (y, x)
                grid[y][x] = " "

    if start is None or goal is None:
        raise ValueError("Maze must contain @ (start) and G (goal)")
    return grid, start, goal


def _draw(stdscr, grid, player, goal, status: str):
    stdscr.erase()
    h = len(grid)
    w = len(grid[0])

    for y in range(h):
        stdscr.addstr(y, 0, "".join(grid[y]))

    py, px = player
    gy, gx = goal
    stdscr.addch(gy, gx, "G")
    stdscr.addch(py, px, "@")

    stdscr.addstr(h + 1, 0, status[: max(0, w - 1)])
    stdscr.refresh()


def _game(stdscr, maze_text: str):
    curses.curs_set(0)
    stdscr.keypad(True)
    curses.noecho()
    curses.cbreak()

    grid, player, goal = _parse_maze(maze_text)
    h = len(grid)
    w = len(grid[0])

    max_y, max_x = stdscr.getmaxyx()
    needed_y = h + 2
    needed_x = w + 1
    if max_y < needed_y or max_x < needed_x:
        stdscr.erase()
        stdscr.addstr(
            0,
            0,
            f"Terminal too small: need at least {needed_x}x{needed_y}, got {max_x}x{max_y}.",
        )
        stdscr.addstr(2, 0, "Resize and run again.")
        stdscr.refresh()
        stdscr.getch()
        return

    status = "Arrow keys to move, q to quit."
    _draw(stdscr, grid, player, goal, status)

    key_to_delta = {
        curses.KEY_UP: (-1, 0),
        curses.KEY_DOWN: (1, 0),
        curses.KEY_LEFT: (0, -1),
        curses.KEY_RIGHT: (0, 1),
        ord("k"): (-1, 0),
        ord("j"): (1, 0),
        ord("h"): (0, -1),
        ord("l"): (0, 1),
    }

    while True:
        ch = stdscr.getch()
        if ch in (ord("q"), ord("Q")):
            return

        d = key_to_delta.get(ch)
        if d is None:
            continue

        dy, dx = d
        py, px = player
        ny, nx = py + dy, px + dx

        if grid[ny][nx] == "#":
            status = "Bump! (q to quit)"
        else:
            player = (ny, nx)
            status = "Arrow keys to move, q to quit."

        if player == goal:
            _draw(stdscr, grid, player, goal, "You win! Press any key.")
            stdscr.getch()
            return

        _draw(stdscr, grid, player, goal, status)


def _choose_difficulty() -> str:
    options = ("easy", "medium", "hard")
    while True:
        choice = input("Choose difficulty (easy/medium/hard) [medium]: ").strip().lower()
        if choice == "":
            return "medium"
        if choice in options:
            return choice
        print("Please type: easy, medium, or hard.")


def main():
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print("This game uses curses and must be run in a real terminal (TTY).")
        print("Try: python -m src.maze.maze")
        return
    difficulty = _choose_difficulty()
    curses.wrapper(_game, MAZES[difficulty])


if __name__ == "__main__":
    main()
