import random
import Othello as o
import Board as b
from tqdm import tqdm

def display_node(node, indent = 0):
    display_indent = ('|    ' * indent) if indent > 0 else ''

    print(display_indent, end='')
    if node.to_play: print("Black's Turn")
    else: print("White's Turn")

    print(f'{display_indent}| visits: {node.visits}')
    print(f'{display_indent}| score: {node.score}')
    print(f'{display_indent}| black: {int.bit_count(node.black)} Tiles')
    print(f'{display_indent}| white: {int.bit_count(node.white)} Tiles')
    print(f'{display_indent}|')

def create_root():
    root = b.Node()
    root.visits = 1
    return root

def random_game(white = 34628173824, black = 68853694464, turn_count = 0):
    if turn_count % 2 == 0: player, opponent = black, white
    else: player, opponent = white, black

    last_turn_pass = False
    while True:
        valid_moves = o.get_valid_move_list(player, opponent)
        if len(valid_moves) == 0: chosen_move = -1
        else: chosen_move = random.choice(valid_moves)

        if chosen_move == -1:
            if last_turn_pass:
                # Game complete! Parsing who is black currently
                #   Returns: White Board, Black Board
                if turn_count % 2 == 0:
                    return o.determine_winner(opponent, player)
                else:
                    return o.determine_winner(player, opponent)
            else:
                last_turn_pass = True
        else:
            player, opponent = o.update_board(chosen_move, player, opponent)
            last_turn_pass = False

        player, opponent = opponent, player
        turn_count += 1

# Node __init__(self, parent=None, white=34628173824, black=68853694464, turn_count=0)
class MonteCarlo:
    def __init__(self, root):
        self.root = root


    def selection(self, node):
        while node.is_explored():
            utc = [(child, child.compute_UCT(2)) for child in node.children]
            max_utc = max([child[1] for child in utc])
            selected_node = random.choice([child for child in utc if child[1] == max_utc])[0]
            node = selected_node
        return node


    def expansion(self, parent):
        child = parent.make_child()
        return child


    def simulation(self, node):
        score = random_game(node.white, node.black, node.turn_count)
        return score


    def backpropagation(self, node, score):
        while node.parent is not None:
            if node.to_play:
                node.score += score
            else:
                node.score += -score
            node.visits += 1
            node = node.parent


    def display(self, node, indent = 0):
        display_node(node, indent)

        for child in node.children:
            self.display(child, indent + 1)

def monte_carlo_tree_search(root = None, iterations = 1000):
    if root is None:
        tree = MonteCarlo(create_root())
    else:
        if root.visits < 1: root.visits = 1
        tree = MonteCarlo(root)

    progress_bar = tqdm(range(iterations))
    for iteration in progress_bar:
        node = tree.selection(tree.root)
        child = tree.expansion(node)
        score = tree.simulation(child)
        tree.backpropagation(child, score)

        if (iteration + 1) % 100 == 0:
            best_move = tree.root.compute_best_score()[1]
            move = f'{chr(best_move % 8 + 65)}{best_move // 8 + 1}' if best_move != -1 else 'pass'
            progress_bar.set_postfix({'Top Move': {move}})

    #DEBUG BELOW: Display entire Tree
    # tree.display(tree.root)
    return tree.root.compute_best_score()


if __name__ == '__main__':
    # Test Code running through 2 MCTS moves
    white, black = 34628173824, 68853694464
    node, move = monte_carlo_tree_search()
    black, white = o.update_board(move, black, white)
    o.disp_game(white, black, False)

    valid_moves = o.get_valid_move_list(white, black)
    move = random.choice(valid_moves)
    white, black = o.update_board(move, white, black)
    o.disp_game(white, black, True)

    new_root = None
    if move in node.available_moves:
        new_root = node.make_child(move)
    else:
        for child in node.children:
            if child.move == move:
                new_root = child
                break

    node, move = monte_carlo_tree_search(new_root)
    black, white = o.update_board(move, black, white)
    o.disp_game(white, black, False)
