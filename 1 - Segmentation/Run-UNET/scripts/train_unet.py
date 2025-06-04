import numpy as np
import torch
import os
from tqdm import tqdm
import sys
import cv2
import csv
import wandb
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir)
    )
)

import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from src.models.discriminator import CNNClassifier
from src.dataloader.dataloaders import MadisonDatasetLabeled
from src.models.unet import BaseUNet
from src.utils.viz_utils import visualize_predictions
from src.utils.args_utils import train_arg_parser
from src.evaluation.segmentation_metrics import dice_coefficient
from src.utils.variable_utils import PLOT_DIRECTORY, TRAINING_LOO, VALIDATION_LOO

UNET_PERFORMANCE_STATISTICS_FOLDER = '/home/miguel/GI/1 - Segmentation/UNet-and-Synthesis-Results/unet_performance_statistics'

# Define the plot_metric function
def plot_metric(x, label, plot_dir, args, metric):
    plt.figure()
    plt.plot(x, label=label)
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.title(f'{metric.capitalize()} Over Epochs')
    plt.legend()

    # Save the plot
    plot_path = os.path.join(plot_dir, args.exp_id, f'{metric}_curve.jpg')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, format='jpg', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()

    print(f'{metric.capitalize()} plot saved to {plot_path}')
    return plot_path    

def parse_path(path):
    filename = os.path.basename(path)  # Get the filename, e.g., 'FD_027_time_1_slice_7_image.png'
    parts = filename.split('_')  # Split by underscores
    subject = parts[0]  # e.g., 'FD_027'
    time_point = parts[2]  # e.g., '1'
    slice_idx = parts[4]  # e.g., '7'
    return subject, time_point, slice_idx

def train_model(model, train_loader, val_loader, optimizer, criterion, device, args, discriminator_model = None, discriminator_lr = 1e-3):
    exp_dir = os.path.join(UNET_PERFORMANCE_STATISTICS_FOLDER, args.exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'model'), exist_ok=True)

    train_loss_history = []
    val_loss_history = []
    dice_coef_history = []
    best_dice_predictions = []  # Store predictions for the best Dice epoch

    best_dice = -float('inf')  # Initialize the best Dice coefficient to a very low value
    patience_counter = 0
    patience_limit = 10
    best_dice_epoch = -1  # Track the best Dice epoch

    if discriminator_model is not None:
        discriminator_model.to(device)
        discriminator_criterion = nn.BCELoss()  
        discriminator_optimizer = torch.optim.Adam(discriminator_model.parameters(), lr=discriminator_lr)

    for epoch in range(args.epoch):
        model.train()
        train_loss = 0.0
        discrim_running_loss = 0.0
        discrim_correct = 0
        discrim_total = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epoch}")):
            images, masks, paths = batch
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

            # Train the discriminator model
            # if discriminator_model is not None:
            # TODO: Acquire labels from the train loader, are the samples real or fake?
            if discriminator_model is not None:
                discriminator_model.train()

                discrim_labels = torch.tensor([1 if 'fake' in path else 0 for path in paths], dtype=torch.float32).to(device)
                discriminator_optimizer.zero_grad()
                discrim_preds = discriminator_model(images).squeeze(1)

                loss_discrim = discriminator_criterion(discrim_preds, discrim_labels)
                loss_discrim.backward()
                discriminator_optimizer.step()

                discrim_running_loss += loss_discrim.item() * images.size(0)
                preds = (discrim_preds >= 0.5).long()
                discrim_correct += (preds == discrim_labels.long()).sum().item()
                discrim_total += discrim_labels.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        train_loss_history.append(train_loss)

        model.eval()

        if discriminator_model is not None:
            discrim_epoch_loss = discrim_running_loss / discrim_total
            discrim_epoch_acc = discrim_correct / discrim_total
            print(f"Discriminator Loss: {discrim_epoch_loss:.4f}, Discriminator Accuracy: {discrim_epoch_acc:.4f}")
            
        val_loss = 0.0
        dice_coefficients = []
        current_epoch_predictions = []  # Store predictions for the current epoch
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                images, masks, paths = batch
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)

                # Loss calculation
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

                # Save predictions to memory (not disk yet)
                for i, path in enumerate(paths):
                    predicted_mask = (torch.sigmoid(outputs[i]) > 0.5).cpu().numpy()
                    predicted_mask = (predicted_mask.squeeze() * 255).astype(np.uint8)

                    # Store predictions
                    current_epoch_predictions.append((predicted_mask, path))

                # Compute Dice coefficient per sample
                for i in range(images.size(0)):
                    dice_score = dice_coefficient(outputs[i], masks[i], threshold=0.1)
                    dice_coefficients.append(dice_score)


        # Average Dice across all validation samples
        dice_mean = np.mean(dice_coefficients)
        dice_coef_history.append(dice_mean)
        val_loss = val_loss / len(val_loader.dataset)
        val_loss_history.append(val_loss)

        print(f"Dice Coefficient for Epoch {epoch}: {dice_mean:.4f}")
        print(f'Epoch {epoch+1}/{args.epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        wandb.log({
            "epoch": epoch,
            'gen_model': args.gen_model,
            'synthetic_real_ratio': args.synthetic_real_ratio,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "dice_coefficient": dice_mean,
            "predictions": [wandb.Image(pred[0], caption=os.path.basename(pred[1])) for pred in current_epoch_predictions]
        })

        # Save the model and predictions only if Dice coefficient improves
        if dice_mean > best_dice:
            best_dice = dice_mean
            best_dice_epoch = epoch
            best_dice_predictions = current_epoch_predictions  # Update best predictions

            save_path = os.path.join(exp_dir, 'model', 'best_model.pt')
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved at epoch {epoch + 1} with Dice coefficient {best_dice:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience_limit:
            print(f"Stopping early after {epoch + 1} epochs due to no improvement in Dice coefficient.")
            break

    # Save predictions for the best Dice epoch to disk
    save_dir = os.path.join(exp_dir, 'predictions')
    os.makedirs(save_dir, exist_ok=True)
    for predicted_mask, original_path in best_dice_predictions:
        # Save the predicted mask with a consistent filename
        original_filename = os.path.basename(original_path).replace("_image.png", "_predicted.png")
        save_path = os.path.join(save_dir, original_filename)
        cv2.imwrite(save_path, predicted_mask)



    print(f"Predictions for the best Dice epoch ({best_dice_epoch + 1}) saved to {save_dir}")

    # Save best Dice epoch info to a text file
    best_dice_path = os.path.join(exp_dir, 'best_dice_epoch.txt')
    with open(best_dice_path, 'w') as f:
        f.write(f"Best Epoch: {best_dice_epoch + 1}\n")
        f.write(f"Dice Coefficient: {best_dice:.4f}\n")

    print(f"Best Dice epoch details saved to {best_dice_path}")

    # Plot and save Dice coefficient curve
    plot_save_path = plot_metric(x=dice_coef_history,
                label="Dice Coefficient",
                plot_dir=PLOT_DIRECTORY,
                args=args,
                metric='dice_coeff')

    # Save the curve to wandb
    wandb.log({
        "dice_coefficient_curve": wandb.Image(plot_save_path),
        "synthetic_real_ratio": args.synthetic_real_ratio,
        "gen_model": args.gen_model
    })

    print("Training completed.")

if __name__ == "__main__":
    args = train_arg_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize the datasets
    print(f"Loading training data from: {args.train_dir}")
    train_dataset = MadisonDatasetLabeled(args.train_dir, augment=True)
    print(f"Loading validation data from: {args.val_dir}")
    val_dataset = MadisonDatasetLabeled(args.val_dir, augment=False)

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)





    import pandas as pd
    from torchvision import io
    data_shapes = []
    for i in tqdm(range(len(train_dataset))):
        img, mask, path = train_dataset[i]
        if img.shape[0] != 1:
            print(f"image {path} with shape {img.shape} is NOT single channel")
        if mask.shape[0] != 1:
            breakpoint()
            print(f"mask {path} with shape {mask.shape} is NOT single channel")
        data_shapes.append((img.shape, mask.shape, path))

    df = pd.DataFrame(data_shapes, columns=['Image Shape', 'Mask Shape', 'Path'])
    df.to_csv('data_shapes.csv', index=False)







    model = BaseUNet(in_channels=1, out_channels=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    discriminator_model = CNNClassifier().to(device) if args.gen_model else None

    timestamp = np.datetime64('now', 's').astype(str).replace(':', '-')

    wandb.init(
        project="gsoc-diffusion-2025",
        name=f"unet_{args.gen_model}_benchmark_{timestamp}"
    )

    train_model(model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                args=args,
                discriminator_model=discriminator_model)
