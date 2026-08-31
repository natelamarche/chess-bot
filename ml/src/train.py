import torch
from tqdm import tqdm
from datasets import load_from_disk
from torch.utils.data import DataLoader
from ml.src.dataset import ChessNNUEDataset, nnue_collate_fn, MAX_EVAL
from ml.src.model import NNUE, loss_function

def main():
    splits = load_from_disk("ml/data/encoded_positions")

    train_set = splits["train"].with_format("torch")
    val_set = splits["validation"].with_format("torch")

    SEED=10
    torch.manual_seed(SEED)

    train_generator = torch.Generator().manual_seed(SEED)

    train_loader = DataLoader(
        train_set,
        batch_size=16_384,
        shuffle=False,
        generator=train_generator,
        num_workers=4,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=16_384,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    ACCUMULATOR_SIZE=128
    model = NNUE(accumulator_size=ACCUMULATOR_SIZE).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = 1e-3
    )

    train_loss_history = []
    val_loss_history = []

    patience = 5
    best_val_loss = float("inf")
    delta = 1e-4
    epochs_without_improvement = 0

    num_epochs = 50

    for epoch in range(num_epochs):
        model.train()
        
        total_loss = torch.zeros((), dtype=torch.float32).to(device)
        total_positions = 0
        
        progress = tqdm(train_loader, desc=f"Epoch: {epoch+1}/{num_epochs} | Training")
        for batch in progress:
            white_features = batch["white_features"].to(device, dtype=torch.int32, non_blocking=True)
            black_features = batch["black_features"].to(device, dtype=torch.int32, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            stms = batch["side_to_move"].to(device, non_blocking=True)
            
            predictions = model(white_features, black_features, stms)
            
            loss = loss_function(predictions, targets)
            
            optimizer.zero_grad(set_to_none=True)
            
            loss.backward()
            
            optimizer.step()
            
            batch_size = targets.size(0)
            
            total_loss += loss.detach() * batch_size
            
            total_positions += batch_size
        
        epoch_loss = total_loss/total_positions
        
        train_loss_history.append(epoch_loss)
        
        model.eval()

        with torch.no_grad():
            total_loss = torch.zeros((), dtype=torch.float32).to(device)
            total_positions = 0
            
            progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} | Validation")
            for batch in progress:
                white_features = batch["white_features"].to(device, dtype=torch.int32, non_blocking=True)
                black_features = batch["black_features"].to(device, dtype=torch.int32, non_blocking=True)
                targets = batch["target"].to(device, non_blocking=True)
                stms = batch["side_to_move"].to(device, non_blocking=True)
                
                predictions = model(white_features, black_features, stms)
                
                loss = loss_function(predictions, targets)
                
                batch_size = targets.size(0)
                
                total_loss += loss * batch_size
                
                total_positions += batch_size
        
        epoch_loss = total_loss/total_positions
        
        val_loss_history.append(epoch_loss)
        
        if epoch_loss < best_val_loss - delta:
            best_val_loss = epoch_loss
            epochs_without_improvement = 0
            
            best_model_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement > patience:
            print("Early stopping")
            break
        
    
        print(f"Epoch: {epoch+1} | Train loss: {train_loss_history[-1]:.4f} | Val loss: {val_loss_history[-1]:.4f}")
        
    model.load_state_dict(best_model_state)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "accumulator_size": ACCUMULATOR_SIZE,
        "feature_schema_version": 1,
        "max_eval": MAX_EVAL,
        "best_val_loss": best_val_loss.item()
    }, "ml/model/model_weights.pth")
    
    print("DONE")
    
if __name__ == "__main__":
    main()