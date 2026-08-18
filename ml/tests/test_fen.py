import unittest

import torch

from ml.src.fen import fen_to_tensor


class FenToTensorTests(unittest.TestCase):
    def test_starting_position(self):
        tensor = fen_to_tensor(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR "
            "w KQkq - 0 1"
        )

        self.assertEqual(tensor.shape, (18, 8, 8))
        self.assertEqual(tensor.dtype, torch.float32)

        expected_board = torch.zeros(12, 8, 8)
        expected_board[0, 1, :] = 1  # Black pawns on rank 7.
        expected_board[1, 0, [1, 6]] = 1
        expected_board[2, 0, [2, 5]] = 1
        expected_board[3, 0, [0, 7]] = 1
        expected_board[4, 0, 3] = 1
        expected_board[5, 0, 4] = 1
        expected_board[6, 6, :] = 1  # White pawns on rank 2.
        expected_board[7, 7, [1, 6]] = 1
        expected_board[8, 7, [2, 5]] = 1
        expected_board[9, 7, [0, 7]] = 1
        expected_board[10, 7, 3] = 1
        expected_board[11, 7, 4] = 1

        torch.testing.assert_close(tensor[:12], expected_board)
        torch.testing.assert_close(tensor[12], torch.zeros(8, 8))
        torch.testing.assert_close(tensor[13:17], torch.ones(4, 8, 8))
        torch.testing.assert_close(tensor[17], torch.zeros(8, 8))

    def test_black_to_move_partial_castling_and_en_passant(self):
        tensor = fen_to_tensor(
            "4k3/8/8/3pP3/8/8/8/4K3 b Kq d6 17 42"
        )

        expected_board = torch.zeros(12, 8, 8)
        expected_board[0, 3, 3] = 1   # Black pawn on d5.
        expected_board[5, 0, 4] = 1   # Black king on e8.
        expected_board[6, 3, 4] = 1   # White pawn on e5.
        expected_board[11, 7, 4] = 1  # White king on e1.
        torch.testing.assert_close(tensor[:12], expected_board)

        torch.testing.assert_close(tensor[12], torch.ones(8, 8))
        torch.testing.assert_close(tensor[13], torch.ones(8, 8))
        torch.testing.assert_close(tensor[14], torch.zeros(8, 8))
        torch.testing.assert_close(tensor[15], torch.zeros(8, 8))
        torch.testing.assert_close(tensor[16], torch.ones(8, 8))

        expected_ep = torch.zeros(8, 8)
        expected_ep[2, 3] = 1  # d6 is row 2, column 3.
        torch.testing.assert_close(tensor[17], expected_ep)

    def test_requires_exactly_six_fen_fields(self):
        invalid_fens = (
            "8/8/8/8/8/8/8/8 w - - 0",
            "8/8/8/8/8/8/8/8 w - - 0 1 extra",
        )

        for fen in invalid_fens:
            with self.subTest(fen=fen):
                with self.assertRaisesRegex(
                    ValueError, r"Invalid FEN: expected 6 fields"
                ):
                    fen_to_tensor(fen)

    def test_rejects_invalid_en_passant_squares(self):
        for square in ("a4", "i3", "A3", "a", "a33"):
            with self.subTest(square=square):
                fen = f"8/8/8/8/8/8/8/8 w - {square} 0 1"
                with self.assertRaisesRegex(
                    ValueError, rf"Invalid en-passant square: {square}"
                ):
                    fen_to_tensor(fen)


if __name__ == "__main__":
    unittest.main()