import torch

def get_piece_from_char(char: str) -> int:
    match char:
        case "p":
            return 0
        case "n":
            return 1
        case "b":
            return 2
        case "r":
            return 3
        case "q":
            return 4
        case "k":
            return 5
        case "P":
            return 6
        case "N":
            return 7
        case "B":
            return 8
        case "R":
            return 9
        case "Q":
            return 10
        case "K":
            return 11
        case _:
            return None

# tensor row 0 = rank 8
# tensor row 7 = rank 1
# tensor column 0 = file a
# tenosr column 7 = file h
def get_board(board_fen: str) -> torch.Tensor:
    board = torch.zeros(12, 8, 8)
    rows = board_fen.split("/")
    r = 0
    for row in rows:
        c = 0
        for column in row:
            piece = get_piece_from_char(column)
            if piece is None:
                c += int(column)
            else:
                board[piece, r, c] = 1
                c += 1
        r += 1

    return board
    
def get_stm(stm_fen: str) -> torch.Tensor:
    return torch.zeros(1, 8, 8) if stm_fen == "w" else torch.ones(1, 8, 8)

def get_castling(castling_fen: str) -> torch.Tensor:
    white_kingside = torch.ones(1, 8, 8) if "K" in castling_fen else torch.zeros(1, 8, 8)
    white_queenside = torch.ones(1, 8, 8) if "Q" in castling_fen else torch.zeros(1, 8, 8)
    black_kingside = torch.ones(1, 8, 8) if "k" in castling_fen else torch.zeros(1, 8, 8)
    black_queenside = torch.ones(1, 8, 8) if "q" in castling_fen else torch.zeros(1, 8, 8)
    
    return torch.cat([white_kingside, white_queenside, black_kingside, black_queenside], dim=0)

def get_ep(ep_fen: str) -> torch.Tensor:
    ep = torch.zeros(1, 8, 8)
    
    if ep_fen == "-":
        return ep
    
    files = "abcdefgh"
    
    if (
        len(ep_fen) != 2
        or ep_fen[0] not in files
        or ep_fen[1] not in "36"
    ):
        raise ValueError(f"Invalid en-passant square: {ep_fen}")
        
    column = files.index(ep_fen[0])
    row = 8 - int(ep_fen[1])
    ep[0, row, column] = 1
    
    return ep
    
# 1-12 -> pieces
# 13 -> stm
# 14 -> white kingside
# 15 -> white queenside
# 16 -> black kingside
# 17 -> black queenside
# 18 -> en passant
def fen_to_tensor(fen: str) -> torch.Tensor:
    
    fields = fen.split()
    if len(fields) != 6:
        raise ValueError(f"Invalid FEN: expected 6 fields, got {len(fields)}")
        
    board_fen = fields[0]
    stm_fen = fields[1]
    castling_fen = fields[2]
    ep_fen = fields[3]
    
    board = get_board(board_fen)
    stm = get_stm(stm_fen)
    castling = get_castling(castling_fen)
    ep = get_ep(ep_fen)
    
    return torch.cat([board, stm, castling, ep], dim=0)