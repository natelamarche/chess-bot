import torch
import torch.nn as nn

from ml.src.model import model, loss_function, optimizer
from ml.src.dataset import train_loader, val_loader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

train_loss_history = []
val_loss_history = []

best_val_loss = float("inf")
patience = 5
min_delta = 1e-4
epochs_without_improvement = 0

best_model_state = None

for epoch in range(50):
    model.train()
    total_loss = torch.zeros((), device=device)
    total_len = 0
    
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        output = model(batch_X)
        loss = loss_function(output, batch_y)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        batch_size = batch_X.size(0)
        total_loss += loss.detach() * batch_size
        total_len += batch_size
    
    train_loss = (total_loss/total_len).item()
    train_loss_history.append(train_loss)

    model.eval() 
    total_loss = torch.zeros((), device=device)
    total_len = 0
    
    with torch.inference_mode():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            ouput = model(batch_X)
            loss = loss_function(ouput, batch_y)
            
            batch_size = batch_X.size(0)
            total_loss += loss * batch_size
            total_len += batch_size
        
    val_loss = (total_loss/total_len).item()
    val_loss_history.append(val_loss)
        
    print(f"Epoch: {epoch + 1} | "
          f"Train Loss: {train_loss_history[-1]:.5f} | "
          f"Val Loss: {val_loss_history[-1]:.5f}"
          )
    
    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        
        best_model_state = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.state_dict().items()
        }
    else: 
        epochs_without_improvement += 1
    
    if epochs_without_improvement >= patience:
        print("Early stopping")
        break
        
model.load_state_dict(best_model_state)