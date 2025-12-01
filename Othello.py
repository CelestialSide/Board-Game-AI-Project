import numpy as np
import random

def g_dex(row, col):
    return row * 8 + col


def set_bit(b, dex):
    return b | (1 << dex)
def read_bit(b, dex):
    return (b >> dex) & 1 == 1
def clear_bit(b, dex):
    return b & ~(1 << dex)


def set_bits_bitboard(board, indices):
    temp = board

    for dex in indices:
        temp = set_bit(temp, dex)

    return temp


def shift_left(b, shift):
    for i in range(shift):
        b = (b >> 1) & 9187201950435737471 # Cancel out right column
    return b
def shift_right(b, shift):
    for i in range(shift):
        b = (b << 1) & 18374403900871474942 # Cancel out left column
    return b
def shift_up(b, shift):
    return b >> shift * 8
def shift_down(b, shift):
    return b << shift * 8


def determine_valid_squares(player, opponent):
    valid_moves = np.uint64(0)

    # Verticals
    valid_moves = valid_moves | (shift_down(opponent, 1) & shift_down(player, 2)) # down moves
    valid_moves = valid_moves | (shift_up(opponent, 1) & shift_up(player, 2)) # up moves

    # Horizontals
    valid_moves = valid_moves | (shift_right(opponent, 1) & shift_right(player, 2)) # right moves
    valid_moves = valid_moves | (shift_left(opponent, 1) & shift_left(player, 2)) # left moves

    # Diagonals
    valid_moves = valid_moves | (shift_up(shift_left(opponent, 1),1)
                                 & shift_up(shift_left(player, 2), 2)) # top-left
    valid_moves = valid_moves | (shift_up(shift_right(opponent, 1), 1)
                                 & shift_up(shift_right(player, 2), 2)) # top-right
    valid_moves = valid_moves | (shift_down(shift_left(opponent, 1), 1)
                                 & shift_down(shift_left(player, 2), 2)) # bottom-left
    valid_moves = valid_moves | (shift_down(shift_right(opponent, 1), 1)
                                 & shift_down(shift_right(player, 2), 2)) # bottom-right

    return valid_moves & ~(player | opponent) # Make sure there are no pieces on the 'valid move'

def gen_move_of_len(move_instructions : list, player, opponent, len_):
    # Len = number of pieces to be flanked
    b = 18446744073709551615 # All true

    temp = move_instructions[0](opponent, 1)

    for k in range(1, len(move_instructions)):
        temp = move_instructions[k](temp, 1)
    b = b & temp

    # We use a for loop to check long flanks
    # to cut down on the use of if statements.
    for i in range(1, len_):
        temp = move_instructions[0](temp, 1)

        for k in range(1, len(move_instructions)):
            temp = move_instructions[k](temp, 1)
        b = b & temp # Check every opponent's piece in 'sandwich'

    # Finally, check to make sure the player has a piece at the end of the 'sandwich'
    temp = move_instructions[0](player, len_+1)
    for k in range(1, len(move_instructions)):
        temp = move_instructions[k](temp, len_+1)
    b = b & temp

    return b  # Verify there is no piece on our valid moves


def shift_boards(boards):
    # Directions: [-9,-8,-7,-1,1,7,8,9]
    boards[0] = (boards[0] >> 9) & 9187201950435737471
    boards[1] = boards[1] >> 8
    boards[2] = (boards[2] >> 7) & 18374403900871474942
    boards[3] = (boards[3] >> 1) & 9187201950435737471
    boards[4] = (boards[4] << 1) & 18374403900871474942
    boards[5] = (boards[5] << 7) & 9187201950435737471
    boards[6] = boards[6] << 8
    boards[7] = (boards[7] << 9) & 18374403900871474942


def advanced_gen_moves(player, opponent):
    # Write a less-generalized, more optimized version of generating moves that cuts out a lot of repeated calculations.
    valid_moves = 0

    # Opponent stubs start out with maximal value (all bits true)
    opponent_stubs = [18446744073709551615 for i in range(
        8)]  # Repeated AND blocks for each direction -> continuous enemy peices -> contains previous work
    opponent_boards = [opponent for i in range(8)]  # Copies of opponent board shifted around
    player_boards = [player for i in range(8)]  # Copies of the player board shifted around

    # Directions: [-9,-8,-7,-1,1,7,8,9]
    # Initialize player boards to be in the correct places for the start of the loop
    shift_boards(player_boards)

    # Now, for the initial loop
    for l in range(1, 7):  # Possible lengths of continuous enemy pieces ~ [1,7]
        # Update opponent boards
        shift_boards(opponent_boards)

        # Update stubs + sum
        stub_sum = 0
        for i in range(8):
            opponent_stubs[i] = opponent_stubs[i] & opponent_boards[i]
            stub_sum += opponent_stubs[i]

        # # End early if all stubs are 0 -> no continuous enemy piece line left
        # if stub_sum == 0:
        #     return valid_moves & ~(player | opponent) # Ensure no piece is on a valid move

        # Shift player boards over and compare with stubs to determine valid moves
        shift_boards(player_boards)

        for i in range(8):
            valid_moves = valid_moves | (opponent_stubs[i] & player_boards[i])

    return valid_moves & ~(player | opponent)


# Note: We don't really need bitboard operations here since we're not keeping track of how the entire board is changing.
# Just using bit shift operations to read bits.

def update_board(move_dex, player, opponent):
    # First, update player board with the move.
    player = set_bit(player, move_dex)

    directions = [-9, -8, -7, -1, 1, 7, 8, 9]

    for direct in directions:
        inbetween = 0
        pos = move_dex + direct

        # Read how many opponent pieces are between the new piece and the other flanking piece
        while pos >= 0 and read_bit(opponent, pos):
            inbetween += 1
            pos = pos + direct

        # If we finish at a piece doing the flanking, convert opponent pieces to player pieces
        if pos >= 0 and read_bit(player, pos):
            # print(inbetween, direct)

            pos = move_dex + direct
            for i in range(inbetween):
                player = set_bit(player, pos)
                opponent = clear_bit(opponent, pos)

                pos = pos + direct

    # Return updated bitboards
    return player, opponent

# Now I just need a way to index the valid moves...
def get_valid_move_list(player, opponent):
    move_b = advanced_gen_moves(player, opponent)

    moves = []
    translated = move_b
    for i in range(64): # Check every bit on the board - starting with the greatest index
        if translated & 1 == 1:
            moves.append(i)

        translated = translated >> 1

    return moves

def disp_game(white, black):
    for i in range(8):
        s = ""

        for k in range(8):
            if read_bit(white, g_dex(i, k)):
                s += "W "
            elif read_bit(black, g_dex(i, k)):
                s += "B "
            else:
                s += "O "

        print(s)


# A game loop
def play_random_game():
    white = set_bits_bitboard(0, [g_dex(3, 3), g_dex(4, 4)])
    black = set_bits_bitboard(0, [g_dex(3, 4), g_dex(4, 3)])

    condition = True
    black_to_play = True
    last_turn_pass = False
    turn = 0
    while condition:

        if black_to_play:
            player = black
            opponent = white
        else:
            player = white
            opponent = black

        valid_moves = get_valid_move_list(player, opponent)
        # valid_moves = [turn] if turn < 64 else []

        if len(valid_moves) > 0:
            chosen = random.choice(valid_moves)
            # print(valid_moves, rand_dex)
            player, opponent = update_board(chosen, player, opponent)

            if black_to_play:
                black = player
                white = opponent
            else:
                black = opponent
                white = player

            last_turn_pass = False
        else:
            if last_turn_pass:
                return white, black  # Game complete!
            else:
                last_turn_pass = True

        black_to_play = not black_to_play
        turn += 1

final_w, final_b = play_random_game()
print("Final Game State:")
disp_game(final_w, final_b)