import random

import Othello
import re
import MonteCarlo as mc
from GUI import Display


def player_turn(black, white, turn, player_color, root, iterations, display):
    match player_color:
        case 'Black':
            if turn % 2:return get_move(white, black, True, root, iterations)
            else:return get_move(black, white, False, root, iterations, display= display)
        case 'White':
            if turn % 2: return get_move(white, black, False, root, iterations, display)
            else: return get_move(black, white, True, root, iterations)
        case _: raise Exception('Invalid Player Color')

def update_node(node, move):
    if node is None:
        node = mc.create_root()


    if move not in [child.move for child in node.children]:
        root = node.make_child(move)
    else:
        root = next(child for child in node.children if child.move == move)
    return root, move

def get_move(player, opponent, montecarlo, root, iterations = 0, display = None):
    valid_moves = Othello.get_valid_move_list(player, opponent)

    if len(valid_moves) < 1: return root, -1

    if montecarlo:
        return mc.monte_carlo_tree_search(root, iterations)
    else:
        while True:
            if display:
                move = display.ask_user_input(valid_moves)
                return update_node(root, move)
            else:
                square = input("Enter a square (Example - A1): ").strip().upper()
                if re.match(r'^[A-H][1-8]$', square):
                    square = list(square)
                    move = 8*(int(square[1]) - 1) + ord(square[0]) - 65
                    if move in valid_moves: return update_node(root, move)
                    else: print("Not a Valid Move")
                else: print("Invalid Input")

def monte_carlo_game(cpu = True, use_gui = False, color = None, primary_iterations = 1000, secondary_iterations = 1000):
    player = black = 68853694464
    opponent = white = 34628173824
    primary_root = None
    secondary_root = None

    turn = 0
    last_turn_pass = False

    display = None
    if use_gui:
        display = Display(500,500)
        display.setup_board([player, opponent])

    while True:
        # Othello.disp_game(white, black, not turn % 2)

        if cpu:
            if turn % 2 == 0:
                primary_root, chosen_move = get_move(black, white, True,
                                                     primary_root, primary_iterations)
                secondary_root, _ = update_node(secondary_root, chosen_move)
            else:
                secondary_root, chosen_move = get_move(white, black, True,
                                                       secondary_root, secondary_iterations)
                primary_root, _ = update_node(primary_root, chosen_move)
        else:
            if display and turn > 0:
                display.set_board_display([player, opponent], turn % 2)
            primary_root, chosen_move = player_turn(black, white, turn, color,
                                                      primary_root, primary_iterations, display)

        if chosen_move == -1:
            if last_turn_pass:
                # Game is Complete!
                while primary_root.parent is not None: primary_root = primary_root.parent
                if cpu:
                    while secondary_root.parent is not None: secondary_root = secondary_root.parent
                    return white, black, (primary_root, secondary_root)

                return white, black, (primary_root, None)
            else:
                last_turn_pass = True
        else:
            player, opponent = Othello.update_board(chosen_move, player, opponent)
            last_turn_pass = False

        player, opponent = opponent, player
        black = player if turn % 2 else opponent
        white = opponent if turn % 2 else player

        turn += 1


def game(use_gui = False):
    player, opponent = 68853694464, 34628173824

    display = None
    if use_gui:
        display = Display(500,500)
        display.setup_board([player, opponent])

    last_turn_pass = False
    turn = 0
    while True:
        if turn % 2 == 0: Othello.disp_game(opponent, player, True)
        else: Othello.disp_game(player, opponent, False)

        valid_moves = Othello.get_valid_move_list(player, opponent)
        if len(valid_moves) > 0:
            if turn > 0:
                display.set_board_display([player, opponent], turn % 2)
            while True:
                if display:
                    move = display.ask_user_input(valid_moves)
                    break
        else: move = -1

        if move == -1:
            if last_turn_pass:
                # Game complete! Parsing who is black currently
                #   Returns: White Board, Black Board
                if turn % 2 == 0:
                    return opponent, player
                else:
                    return player, opponent
            else:
                last_turn_pass = True
        else:
            player, opponent = Othello.update_board(move, player, opponent)
            last_turn_pass = False

        player, opponent = opponent, player
        turn += 1


if __name__ == '__main__':
    winner = 0
    black = 0
    white = 0

    for i in range(20):
        print(f'Game {i + 1}')
        final_w, final_b, tree_root = monte_carlo_game(cpu = True, primary_iterations = 250, secondary_iterations = 500)
        winner += Othello.determine_winner(final_w, final_b)
        black += int.bit_count(final_b)
        white += int.bit_count(final_w)

        print("\nFinal Game State:")
        Othello.disp_game(final_w, final_b, True)

    print(f'Wins: {winner}')
    print(f'Black Tiles: {black}')
    print(f'White Tiles: {white}')
    # match winner:
    #     case 1:
    #         print(f'White has won by {int.bit_count(final_w) - int.bit_count(final_b)} Tiles!')
    #     case -1:
    #         print(f'Black has won by {int.bit_count(final_b) - int.bit_count(final_w)} Tiles!')
    #     case 0:
    #         print(f"It's a Draw!")
