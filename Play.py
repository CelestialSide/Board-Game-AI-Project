import random
import Othello
import re

def new_game():
    return 34628173824, 68853694464

def get_move(player, opponent, cpu):
    valid_moves = Othello.get_valid_move_list(player, opponent)
    if len(valid_moves) == 0: return -1

    if cpu:
        return random.choice(valid_moves)
    else:
        while True:
            square = input("Enter a square (Example - A1): ").strip().upper()
            if re.match(r'^[A-H][1-8]$', square):
                square = list(square)
                move = 8*(int(square[1]) - 1) + ord(square[0]) - 65
                if move in valid_moves: return move
                else: print("Not a Valid Move")
            else: print("Invalid Input")

def game(num_players):
    player, opponent = new_game()

    chosen_move = 0
    last_turn_pass = False
    turn = 0
    while True:
        match num_players:
            case 0: chosen_move = get_move(player, opponent, True)
            case 1:
                if turn % 2 == 0:
                    Othello.disp_game(opponent, player)
                    chosen_move = get_move(player, opponent, False)
                else:
                    chosen_move = get_move(player, opponent, True)
                    print(f'CPU Chooses: {chr(chosen_move % 8 + 65)}{chosen_move // 8 + 1}')
            case 2:
                if turn % 2 == 0: Othello.disp_game(opponent, player)
                else: Othello.disp_game(player, opponent)
                chosen_move = get_move(player, opponent, False)

        if chosen_move == -1:
            if last_turn_pass:
                # Game complete! Parsing who is black currently
                if turn % 2 == 0:
                    return opponent, player
                else:
                    return player, opponent
            else:
                last_turn_pass = True
        else:
            player, opponent = Othello.update_board(chosen_move, player, opponent)
            last_turn_pass = False

        player, opponent = opponent, player
        turn += 1


final_w, final_b = game(2)

print("Final Game State:")
Othello.disp_game(final_w, final_b)
winner, score = Othello.determine_winner(final_w, final_b)
match winner:
    case "W":
        print(f'White has won by {score} tiles')
    case "B":
        print(f'Black has won by {score * -1} tiles')
    case "D":
        print(f"It's a Draw!")