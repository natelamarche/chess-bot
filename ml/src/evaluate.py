import torch
import numpy as np
import matplotlib as plt

from ml.src.dataset import val_loader
from ml.src.model import material_eval
from ml.src.train import model, train_loss_history, val_loss_history

plt.plot(train_loss_history, c=(1,0,0), label="Training Loss")
plt.plot(val_loss_history, c=(0,1,0), label="Validation Loss")
plt.legend()
plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model.to(device)
model.eval()

with torch.inference_mode():
    total_MAE = 0
    total_MSE = 0
    MAE_range = [0,0,0,0]
    MAE_range_len = [0,0,0,0]
    correct_class = 0
    
    total_MAE_mat = 0
    total_MSE_mat = 0
    MAE_range_mat = [0,0,0,0]

    correct_class_mat = 0
    
    total_len = 0
     
    for batch_X, batch_y in val_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        output = model(batch_X).squeeze(-1)
        material = material_eval(batch_X).squeeze(-1)
        batch_y = batch_y.squeeze(-1)

        output_p = convert_to_pawns(output)
        batch_y_p = convert_to_pawns(batch_y)

        batch_size = batch_X.size(0)
        
        error = (output_p - batch_y_p).abs()
        error_mat = (material - batch_y_p).abs()
        
        total_MAE += error.mean().item() * batch_size
        total_MSE += (error**2).mean().item() * batch_size
        correct_class += (torch.sign(output) == torch.sign(batch_y)).sum()
        
        for i, (low, high) in enumerate([(0, 1), (1, 3), (3, 5), (5, 10)]):
            if i == 0:
                mask = (batch_y_p.abs() >= low) & (batch_y_p.abs() <= high)
            else:
                mask = (batch_y_p.abs() > low) & (batch_y_p.abs() <= high)
            
            count = mask.sum().item()
            
            if count > 0:
                MAE_range[i] += error[mask].mean().item() * count 
                MAE_range_mat[i] += error_mat[mask].mean().item() * count
                
                MAE_range_len[i] += count


        total_MAE_mat += error_mat.mean().item() * batch_size
        total_MSE_mat += (error_mat**2).mean().item() * batch_size
        correct_class_mat += (torch.sign(material) == torch.sign(batch_y)).sum()
        
        total_len += batch_size
        
    print(
        f"Model MAE: {total_MAE/total_len:.4f}\n"
        f"Model MSE: {total_MSE/total_len:.4f}\n"
        f"Correct Class: {correct_class/total_len:.4%}\n"
        f"MAE Breakdown\n"
        f"  [0,1]: {MAE_range[0]/MAE_range_len[0]:.4f} on {MAE_range_len[0]} classes\n"
        f"  (1,3]: {MAE_range[1]/MAE_range_len[1]:.4f} on {MAE_range_len[1]} classes\n"
        f"  (3,5]: {MAE_range[2]/MAE_range_len[2]:.4f} on {MAE_range_len[2]} classes\n"
        f"  (5,10]: {MAE_range[3]/MAE_range_len[3]:.4f} on {MAE_range_len[3]} classes\n"

    )
    
    print(
        f"Material MAE: {total_MAE_mat/total_len:.4f}\n"
        f"Material MSE: {total_MSE_mat/total_len:.4f}\n"
        f"Correct Class: {correct_class_mat/total_len:.4%}\n"
        f"MAE Breakdown\n"
        f"  [0,1]: {MAE_range_mat[0]/MAE_range_len[0]:.4f} on {MAE_range_len[0]} classes\n"
        f"  (1,3]: {MAE_range_mat[1]/MAE_range_len[1]:.4f} on {MAE_range_len[1]} classes\n"
        f"  (3,5]: {MAE_range_mat[2]/MAE_range_len[2]:.4f} on {MAE_range_len[2]} classes\n"
        f"  (5,10]: {MAE_range_mat[3]/MAE_range_len[3]:.4f} on {MAE_range_len[3]} classes\n"
    )
    
       
