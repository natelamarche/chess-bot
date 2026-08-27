
train_loader = DataLoader(
    train_set,
    batch_size=4096,
    shuffle=True,
    collate_fn=nnue_collate_fn,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)

val_loader = DataLoader(
    val_set,
    batch_size=4096,
    collate_fn=nnue_collate_fn,
    num_workers=8,
    pin_memory=True
)
