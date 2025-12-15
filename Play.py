import random
import torch
import Othello
import MonteCarlo as mc
from NeuralMonteCarlo import NeuralMonteCarlo
from GUI import Display
from AlphaZeroNetwork import AlphaZeroNet

def update_node(node, move):
    """
    Update Node will create a root Node if not already created. Then it will check if
    the move given was already explored: if it was it will return that node
    along with the move, otherwise it will create a new child node.

    :param node: Node that needs to be updated
    :param move: Game Move
    :return: root, move
    """
    if node is None:
        node = mc.create_root()

    if move not in [child.move for child in node.children]:
        root = node.make_child(move)
    else:
        root = next(child for child in node.children if child.move == move)

    return root

class Player:
    def __init__(self, player_type, board, game_param, initial):
        """
        The Player class holds all relevant information needed to run and update half of the
        current game state. Is built to interact with a separate Player class.

        :param player_type: Determines type of play used (MonteCarlo, Human, Random, ect...)
        :param board: Current integer representation of White/Black board state
        :param game_param: Dictionary with all important network information
        :param initial: Determines whether to use info from primary or secondary
        """

        self.player_type = player_type
        self.board = board
        self.root = None
        self.neural = None

        match player_type:
            case 'carlo': # Monte Carlo Tree Search Algorithm
                if initial:
                    self.carlo_iterations = game_param['primary_carlo_iterations']
                    self.C = game_param['primary_C']
                else:
                    self.carlo_iterations = game_param['secondary_carlo_iterations']
                    self.C = game_param['secondary_C']
            case 'neural': # Neural Network
                if initial:
                    self.neural = NeuralMonteCarlo(network=game_param['primary_network'])
                    self.network_iterations = game_param['primary_network_iterations']
                else:
                    self.neural = NeuralMonteCarlo(network = game_param['primary_network'])
                    self.network_iterations = game_param['secondary_network_iterations']

    def get_move(self, player_2, display):
        """
        Get move takes in all current information necessary to get the next move
            of the game and then returns said move after updating all trees.
        :param player_2: The other player, which includes the other board and root.
        :param display: Needed for GUI function when human is playing
        :return: Move chosen by Player
        """

        move = 0
        # Finds all valid moves. If no moves are available, update both nodes and pass
        valid_moves = Othello.get_valid_move_list(self.board, player_2.board)
        if len(valid_moves) == 0:
            self.root = update_node(self.root, -1)
            player_2.root = update_node(player_2.root, -1)
            return -1

        # Gets next move dependent on player type and then updates root node to prevent de-syncs
        match self.player_type:
            case 'carlo':
                self.root, move = mc.monte_carlo_tree_search(self.root, self.carlo_iterations, self.C)
            case 'player':
                move = display.ask_user_input(valid_moves)
                self.root = update_node(self.root, move)
            case 'neural':
                self.neural.run_iterations(self.network_iterations)
                move = self.neural.get_move_to_play()

                self.root = update_node(self.root, move)
            case 'random':
                move = random.choice(valid_moves)
                self.root = update_node(self.root, move)
            case _: raise Exception("Invalid Player Type!")

        # Updates other player's node and neural tree with move
        player_2.root = update_node(player_2.root, move)
        return move


def game(P1, P2, game_param):
    """
    Runs the game loop with GUI, it should be noted that GUI cannot be turned off!
    :param P1: Inputs type of player for Black
    :param P2: Inputs type of player for White
    :param game_param: All relevant information that the networks for P1 and P2 might use
    :return: Final game state of White and Black's boards along with code for the game
    """

    # Initialize Players
    black = Player(P1, 34628173824, game_param, True)
    white = Player(P2, 68853694464, game_param, False)

    game_code = ''
    turn = 0
    last_turn_pass = False

    # Set up board
    display = Display(500,500)
    display.setup_board([black.board, white.board])

    while True: # Game loop
        if not turn % 2: # Black's turn
            display.set_board_display([black.board, white.board], turn % 2)
            move = black.get_move(white, display)
        else: # White's turn
            display.set_board_display([white.board, black.board], turn % 2)
            move = white.get_move(black, display)

        if move == -1:
            if last_turn_pass:
                # Game is Complete!
                return white.board, black.board, game_code
            else:
                # Pass
                last_turn_pass = True
        else:
            # Valid Move has been given, updating boards
            if not turn % 2: black.board, white.board = Othello.update_board(move, black.board, white.board)
            else: white.board, black.board = Othello.update_board(move, white.board, black.board)
            last_turn_pass = False
        game_code += f' {chr(move % 8 + 65)}{move // 8 + 1}'
        turn += 1

        # Updates Neural Network if present
        if black.player_type == 'neural':
            black.neural.shift_root(white.board, black.board, turn, move)
        if white.player_type == 'neural':
            white.neural.shift_root(white.board, black.board, turn, move)


net_1 = AlphaZeroNet()
net_1.load_state_dict(torch.load('Models/zerodeeptempdeep.pt', map_location=torch.device('cpu')))

net_2 = AlphaZeroNet()
# net_2.load_state_dict(torch.load('Models/zero.pt'))

game_params = {
    # Player 1
    'primary_carlo_iterations': 100,
    'primary_C': 2**.5,

    'primary_network': net_1,
    'primary_network_iterations': 500,

    # Player 2
    'secondary_carlo_iterations': 1000,
    'secondary_C': 2 ** .5,

    'secondary_network': net_2,
    'secondary_network_iterations': 100
}

'''
     Player Keys:
Monte Carlo: carlo
Neural Monte Carlo: neural
Human Player: player
Random Play: random
'''
if __name__ == '__main__':
    black_wins = 0
    white_wins = 0
    for _ in range(20): # Runs Game loop x times
        white_board, black_board, full_game_code = game('neural', 'carlo', game_params)
        winner = Othello.determine_winner(white_board, black_board)
        tiles = abs(int.bit_count(white_board) - int.bit_count(black_board))

        Othello.disp_game(white_board, black_board, True)
        # print(full_game_code)
        match winner:
            case 1:
                print(f'White has won by {tiles} Tiles!')
                white_wins += 1
            case -1:
                print(f'Black has won by {tiles} Tiles!')
                black_wins += 1
            case 0:
                print(f"It's a Draw!")

    print(f'Overall Games:')
    print(f'Black: {black_wins}')
    print(f'White: {white_wins}')