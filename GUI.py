from graphics import * # Package is called 'graphics.py'

import Othello


class Display:
    def __init__(self, height, width):
        self.HEIGHT = height
        self.WIDTH = width
        self.GRID_COUNT = 8 # The number of cells in a row and in a column.
        self.win = GraphWin("Othello", width, height)
        self.player_pieces = []
        self.BAR_WIDTH = 10
        self.bar_spacing = (self.WIDTH - (self.GRID_COUNT + 1) * self.BAR_WIDTH) / self.GRID_COUNT

    def setup_board(self, boards):
        self.win.setBackground(color="green")

        horizontal_grid = [Rectangle
                           (Point(0, self.bar_spacing * i + self.BAR_WIDTH * i),
                            Point(self.WIDTH, self.bar_spacing * i + self.BAR_WIDTH * i + self.BAR_WIDTH))
                           for i in range(self.GRID_COUNT + 1)]

        ## For the bars that are long. Vertical refers to how they are spaced.
        vertical_grid = [Rectangle
                         (Point(self.bar_spacing * i + self.BAR_WIDTH * i,0),
                          Point(self.bar_spacing * i + self.BAR_WIDTH * i + self.BAR_WIDTH, self.HEIGHT))
                         for i in range(self.GRID_COUNT + 1)]

        for i in range(self.GRID_COUNT + 1):
            v_current_rec = vertical_grid[i]
            h_current_rec = horizontal_grid[i]

            v_current_rec.setFill(color="white")
            h_current_rec.setFill(color="white")

            v_current_rec.draw(self.win)
            h_current_rec.draw(self.win)

        self.set_board_display(boards)

    def set_board_display(self, boards, player_num = 2):
        """
        boards[0] = Black board
        boards[1] = White board
        player_num = Int representing who is the player and who is the opponent.
        """

        if player_num == 1:
            poss_moves = Othello.advanced_gen_moves(boards[0], boards[1])
        elif player_num == 2:
            poss_moves = Othello.advanced_gen_moves(boards[1], boards[0])

        # Undraw current player pieces
        if self.player_pieces:
            for i in range(len(self.player_pieces)):
                self.player_pieces[i].undraw()

        circle_spacing = self.bar_spacing / 2
        circle_radius = self.bar_spacing / 2 - 5

        for i in range(pow(self.GRID_COUNT, 2)):
            if Othello.read_bit(boards[0], i):
                x: int = (i % self.GRID_COUNT) * self.bar_spacing + (i % self.GRID_COUNT + 1) * self.BAR_WIDTH + circle_spacing
                y: int = (i // self.GRID_COUNT) * self.bar_spacing + (i // self.GRID_COUNT + 1) * self.BAR_WIDTH + circle_spacing
                center = Point(x, y)
                piece = Circle(center, circle_radius)
                piece.setFill(color="white")
                self.player_pieces.append(piece)
                piece.draw(self.win)

            elif Othello.read_bit(boards[1], i):
                x: int = (i % self.GRID_COUNT) * self.bar_spacing + (i % self.GRID_COUNT + 1) * self.BAR_WIDTH + circle_spacing
                y: int = (i // self.GRID_COUNT) * self.bar_spacing + (i // self.GRID_COUNT + 1) * self.BAR_WIDTH + circle_spacing
                center = Point(x, y)
                piece = Circle(center, circle_radius)
                piece.setFill(color="black")
                self.player_pieces.append(piece)
                piece.draw(self.win)

            elif Othello.read_bit(poss_moves, i):
                x: int = (i % self.GRID_COUNT) * self.bar_spacing + (i % self.GRID_COUNT + 1) * self.BAR_WIDTH + circle_spacing
                y: int = (i // self.GRID_COUNT) * self.bar_spacing + (i // self.GRID_COUNT + 1) * self.BAR_WIDTH + circle_spacing
                center = Point(x, y)
                piece = Circle(center, circle_radius)
                piece.setFill(color="green")
                self.player_pieces.append(piece)
                piece.draw(self.win)

        valid_list = Othello.get_valid_move_list(boards[1], boards[0])
        print(self.ask_user_input(valid_list))

    def is_valid(self, valid_moves, chosen_spot):
        for v in valid_moves:
            if v == chosen_spot:
                return True

        return False

    def ask_user_input(self, valid_moves):
        print("Click a valid spot")
        while True:
            user_input = self.mouse_to_bit(self.win.getMouse())
            if self.is_valid(valid_moves, user_input):
                return user_input
            else:
                print("Not a valid move.")

    def mouse_to_bit(self, mouse_point):
        x = (mouse_point.getX() - self.BAR_WIDTH)//(self.bar_spacing + self.BAR_WIDTH)
        y = (mouse_point.getY() - self.BAR_WIDTH)//(self.bar_spacing + self.BAR_WIDTH)

        return x + y * self.GRID_COUNT

display = Display(500, 500)
display.setup_board([68853694464, 34628173824])
display.setup_board([34628173824, 68853694464])
