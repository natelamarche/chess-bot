import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_from_disk
from ml.src.model import NNUE
from ml.src.dataset import ChessNNUEDataset, nnue_collate_fn

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = load_from_disk("ml/data/encoded_positions")
    
    test_set = splits["test"].with_format("torch")

    test_loader = DataLoader(
        test_set,
        batch_size=4096,
        num_workers=8,
        pin_memory=True
    )

    model = NNUE()
    model.load_state_dict(torch.load("ml/model/model_weights.pth", map_location=device)["model_state"])
    model.to(device)

    model.eval()

    with torch.inference_mode():
        total_MAE = 0
        total_MSE = 0
        
        MAE_range = [0,0,0,0]
        MAE_range_len = [0,0,0,0]
        
        correct_class = 0    
        correct_close_class = 0
        
        total_len = 0
        close_len = 0
        
        progress = tqdm(test_loader, desc="Evaluating")
        for batch in progress:
            white_features = batch["white_features"].to(device, dtype=torch.int32, non_blocking=True)
            black_features = batch["black_features"].to(device, dtype=torch.int32, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            stms = batch["side_to_move"].to(device, non_blocking=True)
            
            predictions = model(white_features, black_features, stms)

            batch_size = targets.size(0)
            
            error = (predictions - targets).abs()
            
            total_MAE += error.mean().item() * batch_size
            total_MSE += (error**2).mean().item() * batch_size
            correct_class += (torch.sign(targets) == torch.sign(predictions)).sum()
            
            for i, (low, high) in enumerate([(0, 1), (1, 3), (3, 5), (5, 10)]):
                if i == 0:
                    mask = (targets.abs() >= low) & (targets.abs() <= high)
                else:
                    mask = (targets.abs() > low) & (targets.abs() <= high)
                
                count = mask.sum().item()
                
                if count > 0:
                    MAE_range[i] += error[mask].mean().item() * count                 
                    MAE_range_len[i] += count
            
            close_mask = targets.abs() <= 1
            
            correct_close_class += ((predictions[close_mask] > 0) == (targets[close_mask] > 0)).sum()
            
            close_len += close_mask.sum()
            
            total_len += batch_size
            
        print(
            f"Model MAE: {total_MAE/total_len:.4f}\n"
            f"Model MSE: {total_MSE/total_len:.4f}\n"
            f"Correct Class: {correct_class/total_len:.4%}\n"
            f"Close Correct Class (<1): {correct_close_class/close_len:.4%}\n"
            f"MAE Breakdown\n"
            f"  [0,1]: {MAE_range[0]/MAE_range_len[0]:.4f} on {MAE_range_len[0]} classes\n"
            f"  (1,3]: {MAE_range[1]/MAE_range_len[1]:.4f} on {MAE_range_len[1]} classes\n"
            f"  (3,5]: {MAE_range[2]/MAE_range_len[2]:.4f} on {MAE_range_len[2]} classes\n"
            f"  (5,10]: {MAE_range[3]/MAE_range_len[3]:.4f} on {MAE_range_len[3]} classes\n"
        )       
        
if __name__ == "__main__":
    main()