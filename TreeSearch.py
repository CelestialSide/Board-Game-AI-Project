class TreeNode:
    def __init__(self, data):
        self.data = data
        self.children = []
        self.parent = None
        self.score = 0
        self.traversals = 0

    def get_data(self):
        return self.data

    def get_score(self):
        return self.score

    def set_score(self, s):
        self.score = s

    def add_traversal(self):
        self.traversals += 1

    def get_traversals(self):
        return self.traversals

    def add_child(self, child_node):
        child_node.parent = self
        self.children.append(child_node)

    def get_level(self):
        level = 0
        p = self.parent
        while p:
            level += 1
            p = p.parent
        return level

    def print_tree(self):
        prefix = "  " * self.get_level() + "|-- " if self.parent else ""
        print(prefix + str(self.data))
        if self.children:
            for child in self.children:
                child.print_tree()

    def MCTS(self, p = 2) -> int:
        board = self.get_data()
        if not self.parent:
            moves = open_spots(board)
            for m in moves:
                board[m] = p
                child = TreeNode(board)
                self.add_child(child)
                self.add_traversal()

def score_board(board_state):
    if board_state == -1 or board_state == 0:
        return 0

    return -(board_state * 2 - 3)

def open_spots(board):
    open_spots = []
    for i in range(len(board)):
        if board[i] == 0:
            open_spots.append(i)

    return open_spots

def game_over(board):
    prev_spot = board[0]
    for i in range(1, len(board)):
        if not prev_spot == 0 and board[i] == prev_spot:
            return board[i]
        else:
            prev_spot = board[i]
    if len(open_spots(board)) == 0:
        return -1
    return 0

def ask_for_move(board) -> int:
    print("Input move location(ex. 0 for the first spot): ")
    user_input = input()
    poss_moves = open_spots(board)
    while not validate_input(user_input, poss_moves):
        print("Input move location(ex. 0 for the first spot): ")
        user_input = input()

    return int(user_input)

def validate_input(i, pm):
    for m in pm:
        if i == str(m):
            return True
    return False

if __name__ == "__main__":
    game_board = [0,0,0,0]
    game_state = game_over(game_board)
    p = 1
    while game_state == 0:
        root = TreeNode(game_board)
        if p == 1:
            move = ask_for_move(game_board)
            game_board[move] = p
        else:
            move = root.MCTS()

        print(game_board)
        p = 3 - p