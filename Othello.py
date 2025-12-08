def set_bit(b, dex):
    return b | (1 << dex)
def read_bit(b, dex):
    return (b >> dex) & 1 == 1
def clear_bit(b, dex):
    return b & ~(1 << dex)


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
    opponent_stubs = [18446744073709551615 for _ in range(
        8)]  # Repeated AND blocks for each direction -> continuous enemy pieces -> contains previous work
    opponent_boards = [opponent for _ in range(8)]  # Copies of opponent board shifted around
    player_boards = [player for _ in range(8)]  # Copies of the player board shifted around

    # Directions: [-9,-8,-7,-1,1,7,8,9]
    # Initialize player boards to be in the correct places for the start of the loop
    shift_boards(player_boards)

    # Now, for the initial loop
    for l in range(1, 7):  # Possible lengths of continuous enemy pieces ~ [1,7)
        # Update opponent boards
        shift_boards(opponent_boards)

        # Update stubs + sum
        stub_sum = 0
        for i in range(8):
            opponent_stubs[i] = opponent_stubs[i] & opponent_boards[i]
            stub_sum += opponent_stubs[i]

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
    col_mask = [-1, 0, 1, -1, 1, -1, 0, 1]

    for k in range(8):
        direct = directions[k]
        mask = col_mask[k]

        inbetween = 0
        pos = move_dex + direct

        # Read how many opponent pieces are between the new piece and the other flanking piece
        while pos >= 0 and read_bit(opponent, pos) and not ((pos % 8 == 0 and mask > 0) or (pos % 8 == 7 and mask < 0)):
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

def disp_game(white, black, blacks_move):
    if blacks_move: moves = advanced_gen_moves(black, white)
    else: moves = advanced_gen_moves(white, black)

    print("\n  A B C D E F G H")
    for i in range(8):
        s = ""
        print(i + 1, end = ' ')
        for k in range(8):
            if read_bit(white, 8*i + k):
                s += f'{chr(9679)} '
            elif read_bit(black, 8*i + k):
                s += f'{chr(9675)} '
            elif read_bit(moves, 8*i + k):
                s += "X "
            else:
                s += f'{chr(9633)} '

        print(s)

def determine_winner(white, black):
    winner = int.bit_count(white) - int.bit_count(black)

    if winner == 0:
        return 0 # Draw!
    elif winner > 0:
        return 1 # White wins!
    else:
        return -1 # Black wins!


if __name__ == '__main__':

    white = 0
    black = 0

    black = set_bit(black, 24)
    black = set_bit(black, 23)
    black = set_bit(black, 30)
    white = set_bit(white, 25)
    white = set_bit(white, 38)

    disp_game(white, black, white)
    print()

    white, black = update_board(22, white, black)

    disp_game(white, black, black)