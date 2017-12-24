import enum
from functools import lru_cache
import math
import sys
import traceback
import tty

# run `python checkers.py`, then use the inputs 17,26 40,33 26,40

SIDE_SIZE = 8
BOARD_SIZE = SIDE_SIZE ** 2
ACTION_SPACE = BOARD_SIZE ** 2

@enum.unique
class Player(enum.Enum):
    WHITE = 0
    BLACK = 1

    def __invert__(self):
        return Player.BLACK if self is Player.WHITE else Player.WHITE


@enum.unique
class Tile(enum.Enum):
    EMPTY           = 0
    WHITE_CHECKER   = 1
    WHITE_KING      = 2
    BLACK_CHECKER   = 3
    BLACK_KING      = 4


def board_new():
    board = []
    for i in range(3 * SIDE_SIZE):
        board.append(Tile.WHITE_CHECKER.value * (i + math.floor(i / SIDE_SIZE)) % 2)
    for i in range(2 * SIDE_SIZE):
        board.append(0)
    for i in range(3 * SIDE_SIZE):
        board.append(Tile.BLACK_CHECKER.value * ((i + 1 + math.floor(i / SIDE_SIZE)) % 2))
    return board


def board_print(board):
    for i, tile in enumerate(board):
        if tile == Tile.EMPTY.value:
            sys.stdout.write(' ')
        elif tile == Tile.WHITE_CHECKER.value:
            sys.stdout.write('o')
        elif tile == Tile.WHITE_KING.value:
            sys.stdout.write('O')
        elif tile == Tile.BLACK_CHECKER.value:
            sys.stdout.write('x')
        elif tile == Tile.BLACK_KING.value:
            sys.stdout.write('X')
        if i % SIDE_SIZE == SIDE_SIZE - 1:
            sys.stdout.write('\n')
    sys.stdout.flush()


class Bool:

    def __init__(self, fn, *args):
        self.fn = fn
        self.args = args
        self.or_fns = []

    def __and__(self, operand):
        if not self.value:
            return self

        return operand

    def __or__(self, operand):
        if self.value:
            return self

        operand.log_or(self.fn.__name__)
        return operand

    def log_or(self, name):
        self.or_fns.append(name)

    @property
    def value(self):
        if not hasattr(self, '_value'):
            self._value = self.fn(*self.args)
        return self._value

    @property
    def reason(self):
        return '({}) returned false'.format(' | '.join(self.or_fns))

    def done(self):
        if not self.value:
            self.or_fns.append(self.fn.__name__)


def tile_get_player(tile):
    if tile == Tile.WHITE_CHECKER.value or tile == Tile.WHITE_KING.value:
        return Player.WHITE
    elif tile == Tile.BLACK_CHECKER.value or tile == Tile.BLACK_KING.value:
        return Player.BLACK
    else:
        return None


def move_source_is_player(board, source, player):
    return player == tile_get_player(board[source])


def move_dest_is_open(board, dest):
    return board[dest] == Tile.EMPTY.value


def move_is_diagonal(source, dest):
    dist = math.fabs(dest - source)
    return dist == SIDE_SIZE - 1 or dist == SIDE_SIZE + 1


def move_is_in_board(source, dest):
    if dest < 0 or dest > BOARD_SIZE:
        return False

    a = source % SIDE_SIZE
    b = dest % SIDE_SIZE
    if a == 0: # must move to the right
        return b == 1
    elif a == SIDE_SIZE - 1: # must move to the left
        return b == SIDE_SIZE - 2
    else:
        return True


def move_is_forward(player, source, dest):
    if player == Player.WHITE:
        return dest > source
    else:
        return dest < source


def move_source_is_king(board, source):
    return board[source] == Tile.WHITE_KING.value or board[source] == Tile.BLACK_KING.value


@lru_cache(maxsize=1)
def jumped_player_position(source, dest):
    return int(source + (dest - source) / 2)


#@lru_cache(maxsize=1)
def move_jumps(board, player, source, dest):
    dist = math.fabs(dest - source)
    return ((dist == 2 * (SIDE_SIZE - 1) or dist == 2 * (SIDE_SIZE + 1))
            and board[jumped_player_position(source, dest)] == ~player)


def move_grants_king(player, dest):
    if player is Player.WHITE:
        return dest >= BOARD_SIZE - SIDE_SIZE
    else:
        return dest < SIDE_SIZE


class MoveError(Exception):
    pass


def move_validate(board, player, source, dest):
    result = (Bool(move_source_is_player, board, source, player)
            & Bool(move_is_in_board, source, dest)
            & Bool(move_dest_is_open, board, dest)
            & (Bool(move_is_diagonal, source, dest) | Bool(move_jumps, board, player, source, dest))
            & (Bool(move_is_forward, player, source, dest) | Bool(move_source_is_king, board, source)))
    result.done()
    if not result.value:
        raise MoveError(result.reason)


def move(board, player, source, dest):
    move_validate(board, player, source, dest)
    new_board = board[:]
    if move_jumps(board, player, source, dest):
        new_board[jumped_player_position(source, dest)] = Tile.EMPTY.value
    new_board[dest] = new_board[source]
    new_board[source] = Tile.EMPTY.value
    if move_grants_king(player, dest):
        new_board[dest] = Tile.WHITE_KING.value if player is Player.WHITE else Tile.BLACK_KING.value
    return new_board


class Game:

    def __init__(self, logging=False):
        self.reset()
        self.logging = logging

    def reset(self):
        self.log = []
        self.board = board_new()
        self.turn = Player.WHITE

    def move(self, source, dest):
        if self.logging:
            self.log.append('Player {} moving from {} to {}'.format(self.turn.name, source, dest))
        try:
            self.board = move(self.board, self.turn, source, dest)
            self.turn = ~self.turn
            return True
        except MoveError as e:
            if self.logging:
                self.log.append(str(e))
            return False

    def is_over(self):
        player = None
        for tile in self.board:
            if not player:
                if tile != Tile.EMPTY.value:
                    player = tile_get_player(tile)
                    continue
            elif player != tile_get_player(tile):
                return False
        return True


def game_is_over(board):
    first = None
    for square in board:
        if square != Tile.EMPTY.value:
            if first and first != square:
                return False
            else:
                first = square
    return True


if __name__ == "__main__":
    player = Player.WHITE
    board = board_new()
    message = ''
    while not game_is_over(board):
        sys.stdout.write(u'\u001b[2J')
        print(message)
        message = ''
        board_print(board)
        raw = input('input move for player {}: '.format(player))
        try:
            [source, dest] = raw.split(',')
            source = int(source)
            dest = int(dest)
        except:
            message = 'invalid move: must be formatted as "a,b"'
            continue

        try:
            board = move(board, player, source, dest)
        except MoveError as e:
            message = str(e)
        except:
            message = traceback.format_exc()
        else:
            player = ~player
