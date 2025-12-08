import string

from graphics import * # Package is called 'graphics.py'
from networkx.algorithms.bipartite.basic import color

import Othello


class Display:
    def __init__(self, height, width):
        self.HEIGHT = height
        self.WIDTH = width
        self.LABEL_OFFSET = 20
        self.GRID_COUNT = 8 # The number of cells in a row and in a column.
        self.win = GraphWin("Othello", width + self.LABEL_OFFSET, height + self.LABEL_OFFSET)
        self.player_pieces = []
        self.BAR_WIDTH = 10
        self.bar_spacing = (self.WIDTH - (self.GRID_COUNT + 1) * self.BAR_WIDTH) / self.GRID_COUNT

    def setup_board(self, boards):
        self.win.setBackground(color="green")

        horizontal_grid = [Rectangle
                           (Point(0 + self.LABEL_OFFSET, self.bar_spacing * i + self.BAR_WIDTH * i + self.LABEL_OFFSET),
                            Point(self.WIDTH + self.LABEL_OFFSET, self.bar_spacing * i + self.BAR_WIDTH * i + self.BAR_WIDTH + self.LABEL_OFFSET))
                           for i in range(self.GRID_COUNT + 1)]

        ## For the bars that are long. Vertical refers to how they are spaced.
        vertical_grid = [Rectangle
                         (Point(self.bar_spacing * i + self.BAR_WIDTH * i + self.LABEL_OFFSET,0 + self.LABEL_OFFSET),
                          Point(self.bar_spacing * i + self.BAR_WIDTH * i + self.BAR_WIDTH + self.LABEL_OFFSET, self.HEIGHT + self.LABEL_OFFSET))
                         for i in range(self.GRID_COUNT + 1)]

        for i in range(self.GRID_COUNT + 1):
            v_current_rec = vertical_grid[i]
            h_current_rec = horizontal_grid[i]

            v_current_rec.setFill(color="white")
            h_current_rec.setFill(color="white")

            v_current_rec.draw(self.win)
            h_current_rec.draw(self.win)

            upper_case = string.ascii_uppercase
            char_offset = (self.bar_spacing + self.BAR_WIDTH) / 2
            letter_x = self.LABEL_OFFSET + char_offset * 2 * i + char_offset
            letter_y = self.LABEL_OFFSET / 2
            num_x = self.LABEL_OFFSET / 2
            num_y = self.LABEL_OFFSET + char_offset * 2 * i + char_offset

            current_char = Text(Point(letter_x, letter_y), upper_case[i])
            current_num = Text(Point(num_x, num_y), str(i + 1))

            current_char.setTextColor("white")
            current_num.setTextColor("white")

            current_char.draw(self.win)
            current_num.draw(self.win)

        self.set_board_display(boards)

    def set_board_display(self, boards, player_num = 0):
        """
        boards[0] = current player board
        boards[1] = opponent board
        player_num = Int representing who is the player and who is the opponent.
        """
        if player_num:
            white = boards[0]
            black = boards[1]
        else:
            white = boards[1]
            black = boards[0]
        poss_moves = Othello.advanced_gen_moves(boards[0], boards[1])

        # Undraw current player pieces
        if self.player_pieces:
            for i in range(len(self.player_pieces)):
                self.player_pieces.pop().undraw()

        circle_spacing = self.bar_spacing / 2
        circle_radius = self.bar_spacing / 2 - 5

        for i in range(pow(self.GRID_COUNT, 2)):
            x: int = self.LABEL_OFFSET
            y: int = self.LABEL_OFFSET
            if Othello.read_bit(white, i):
                x += (i % self.GRID_COUNT) * self.bar_spacing + (i % self.GRID_COUNT + 1) * self.BAR_WIDTH + circle_spacing
                y += (i // self.GRID_COUNT) * self.bar_spacing + (i // self.GRID_COUNT + 1) * self.BAR_WIDTH + circle_spacing
                center = Point(x, y)
                piece = Circle(center, circle_radius)
                piece.setFill(color="white")
                self.player_pieces.append(piece)
                piece.draw(self.win)

            elif Othello.read_bit(black, i):
                x += (i % self.GRID_COUNT) * self.bar_spacing + (i % self.GRID_COUNT + 1) * self.BAR_WIDTH + circle_spacing
                y += (i // self.GRID_COUNT) * self.bar_spacing + (i // self.GRID_COUNT + 1) * self.BAR_WIDTH + circle_spacing
                center = Point(x, y)
                piece = Circle(center, circle_radius)
                piece.setFill(color="black")
                self.player_pieces.append(piece)
                piece.draw(self.win)

            elif Othello.read_bit(poss_moves, i):
                x += (i % self.GRID_COUNT) * self.bar_spacing + (i % self.GRID_COUNT + 1) * self.BAR_WIDTH + circle_spacing
                y += (i // self.GRID_COUNT) * self.bar_spacing + (i // self.GRID_COUNT + 1) * self.BAR_WIDTH + circle_spacing
                center = Point(x, y)
                piece = Circle(center, circle_radius)
                piece.setFill(color="green")
                self.player_pieces.append(piece)
                piece.draw(self.win)

    def is_valid(self, valid_moves, chosen_spot):
        for v in valid_moves:
            if v == chosen_spot:
                return True

        return False

    def ask_user_input(self, valid_moves) -> int:
        print("Click a valid spot")
        while True:
            user_input = self.mouse_to_bit(self.win.getMouse())
            if self.is_valid(valid_moves, user_input):
                return int(user_input)
            else:
                print("Not a valid move.")

    def mouse_to_bit(self, mouse_point):
        x = (mouse_point.getX() - self.BAR_WIDTH)//(self.bar_spacing + self.BAR_WIDTH)
        y = (mouse_point.getY() - self.BAR_WIDTH)//(self.bar_spacing + self.BAR_WIDTH)

        return x + y * self.GRID_COUNT

    def close(self):
        self.win.close()
