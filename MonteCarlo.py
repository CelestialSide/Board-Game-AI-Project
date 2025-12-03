import numpy as np
import math
import random


def game():
    board = np.zeros(4)
    player = 1
    cpu_one_in_a_row = False
    mc_one_in_a_row = False

    for _ in range(4):
        valid = board == 0

        if player == 1:
            available = random.choice(np.where(valid)[0])
        else:
            board[random.choice(np.where(valid)[0])] = -1

        for tile in board:
            if tile == -1:
                if cpu_one_in_a_row: return -1
                else:
                    cpu_one_in_a_row = True
                    mc_one_in_a_row = False
            if tile == 1:
                if mc_one_in_a_row: return 1
                else:
                    mc_one_in_a_row = True
                    cpu_one_in_a_row = False
            if tile == 0:
                cpu_one_in_a_row = False
                mc_one_in_a_row = False


    print("Draw")
    return 0

if __name__ == "__main__":
    print(game())