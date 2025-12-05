from graphics import * # Package is called 'graphics.py'
from joblib.externals.loky.backend.fork_exec import fork_exec

import Othello
import Play


class Display:
    def __init__(self, height, width):
        self.HEIGHT = height
        self.WIDTH = width
        self.GRID_COUNT = 8 # The number of cells in a row and in a column.
        self.win = GraphWin("Othello", width, height)
        self.player_pieces = []

    def setup_board(self, boards):
        self.win.setBackground(color_rgb(15,102,11))

        BAR_WIDTH = 10
        bar_spacing = (self.WIDTH - (self.GRID_COUNT + 1) * BAR_WIDTH) / self.GRID_COUNT

        horizontal_grid = [Rectangle
                           (Point(0, bar_spacing * i + BAR_WIDTH * i),
                            Point(self.WIDTH, bar_spacing * i + BAR_WIDTH * i + BAR_WIDTH))
                           for i in range(self.GRID_COUNT + 1)]

        ## For the bars that are long. Vertical refers to how they are spaced.
        vertical_grid = [Rectangle
                         (Point(bar_spacing * i + BAR_WIDTH * i,0),
                          Point(bar_spacing * i + BAR_WIDTH * i + BAR_WIDTH, self.HEIGHT))
                         for i in range(self.GRID_COUNT + 1)]

        for i in range(self.GRID_COUNT + 1):
            v_current_rec = vertical_grid[i]
            h_current_rec = horizontal_grid[i]

            v_current_rec.setFill(color="white")
            h_current_rec.setFill(color="white")

            v_current_rec.draw(self.win)
            h_current_rec.draw(self.win)

        self.set_board_display(boards)

    def set_board_display(self, boards):
        """
        boards[0] = White board
        boards[1] = Black board
        """

        # Undraw current player pieces
        if self.player_pieces:
            for i in range(len(self.player_pieces)):
                self.player_pieces[i].undraw()

        BAR_WIDTH = 10
        bar_spacing = (self.WIDTH - (self.GRID_COUNT + 1) * BAR_WIDTH) / self.GRID_COUNT
        circle_spacing = bar_spacing / 2
        circle_radius = bar_spacing / 2 - 10

        for i in range(pow(self.GRID_COUNT, 2)):
            if Othello.read_bit(boards[0], i):
                x: int = i % self.GRID_COUNT
                y: int = i // self.GRID_COUNT
                center = Point(x, y)
                piece = Circle(center, circle_radius)
                piece.setFill(color="white")
                self.player_pieces.append(piece)
                piece.draw(self.win)

            elif Othello.read_bit(boards[1], i):
                x: int = i % self.GRID_COUNT * bar_spacing
                y: int = i // self.GRID_COUNT * bar_spacing
                center = Point(x, y)
                piece = Circle(center, circle_radius)
                piece.setFill(color="black")
                self.player_pieces.append(piece)
                piece.draw(self.win)

        self.win.getMouse()

display = Display(500, 500)
display.setup_board(Play.new_game())