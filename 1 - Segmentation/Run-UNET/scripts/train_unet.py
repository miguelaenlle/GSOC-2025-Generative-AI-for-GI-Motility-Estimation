import numpy as np
import torch
import os
from tqdm import tqdm
import sys
import cv2
import csv
import wandb
import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold
from torch import amp
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir)
    )
)

from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from src.models.discriminator import CNNClassifier
from src.dataloader.dataloaders import MadisonDatasetLabeled
from src.models.unet import BaseUNet
from src.utils.viz_utils import visualize_predictions
from src.utils.args_utils import train_arg_parser
from src.evaluation.segmentation_metrics import dice_coefficient
from src.utils.variable_utils import PLOT_DIRECTORY, TRAINING_LOO, VALIDATION_LOO
from sklearn.model_selection import KFold
import segmentation_models_pytorch as smp

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

def train_model(
    # model, 
    train_dataset,
    val_loader, 
    # optimizer, 
    # scheduler,
    # criterion, 
    device, 
    args, 
    discriminator_model = None, 
    discriminator_lr = 1e-3,
    kfold_cv_enabled = False
):
    exp_dir_base = os.path.join(UNET_PERFORMANCE_STATISTICS_FOLDER, args.exp_id)
    os.makedirs(exp_dir_base, exist_ok=True)

    patience_counter = 0
    patience_limit = 10

    if discriminator_model is not None: 
        discriminator_model.to(device)
        discriminator_criterion = nn.BCELoss()  
        discriminator_optimizer = torch.optim.Adam(discriminator_model.parameters(), lr=discriminator_lr)

    # kf = KFold(n_splits=5, shuffle=True, random_state=42)

    print('Save directory:', exp_dir_base)

    # fold_train_cv_val_indices = []

    # Iterate over the raw dataset

    subject_indices = {}

    for idx, (image, mask, path) in enumerate(train_dataset):
        filename = os.path.basename(path)
        subject = filename[:filename.find('-slice')]
        if subject not in subject_indices:
            subject_indices[subject] = []
        subject_indices[subject].append(idx)

    subject_keys = list(subject_indices.keys())

    if not kfold_cv_enabled:
        subject_keys = [subject_keys[0]]  # Use only the first subject for training if kfold_cv is not enabled


    for cv_val_subject in subject_keys:
        if kfold_cv_enabled:
            cv_val_idx = subject_indices[cv_val_subject]
            train_idx = []
            for subject, indices in subject_indices.items():
                if subject != cv_val_subject:
                    train_idx.extend(indices)
            print('Train indices:', train_idx, '\nCV Val indices:', cv_val_idx)
            fold = cv_val_subject
        else:
            fold = 1
        # for fold, (train_idx, cv_val_idx) in enumerate(kf.split(range(len(train_dataset))), 1):

        train_loss_history = []
        cv_loss_history = []
        val_loss_history = []

        train_dice_coefficient_history = []
        cv_dice_coefficient_history = []
        val_dice_coef_history = []
        val_dice_coef_history_std = []
        dice_coefficients = [] # Will be the final dice coefficient array upon completion of the loop
        best_dice = -float('inf')  # Initialize the best Dice coefficient to a very low value
        best_dice_epoch = -1  # Track the best Dice epoch
        exp_dir = os.path.join(exp_dir_base, f'fold_{fold}')
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, 'model'), exist_ok=True)
        if kfold_cv_enabled:
            print(f"Training on fold {fold}: train={len(train_idx)}  val={len(cv_val_idx)}")

        if kfold_cv_enabled:
            train_subset = Subset(train_dataset, train_idx)
            cv_val_subset   = Subset(train_dataset, cv_val_idx)
            train_loader = DataLoader(train_subset, batch_size=args.bs, shuffle=True, num_workers=4)
            cv_val_loader   = DataLoader(cv_val_subset,   batch_size=args.bs, shuffle=False, num_workers=4)
        else:
            train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4)
            cv_val_loader = None
    
        if args.unet_architecture == 'uwm-unet':
            model = smp.Unet(
                encoder_name='efficientnet-b1',      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=1,        # model output channels (number of classes in your dataset)
                activation=None,
            )
            model = model.to(device)
        else:
            model = BaseUNet(in_channels=1, out_channels=1).to(device)

        print('Learning rate:', args.lr)

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        criterion = nn.BCEWithLogitsLoss()

        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(30000/args.bs*args.epoch)+50, 
            eta_min=1e-6
        )
        
        best_train_loss = 0.0
        best_train_loss_dice_coefficients = []
        current_epoch_predictions = []

        for epoch in tqdm(range(args.epoch)):
            model.train()
            train_loss = 0.0

            discrim_running_loss = 0.0
            discrim_correct = 0
            discrim_total = 0

            discrim_correct_fake = 0
            discrim_total_fake = 0
            discrim_correct_real = 0
            discrim_total_real = 0

            train_dice_coefficients = []
            scaler = amp.GradScaler()
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epoch}")):
                images, masks, paths = batch
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, masks)
                scaler.scale(loss).backward()

                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
                # optimizer.step()
                train_loss += loss.item() * images.size(0)

                for i in range(images.size(0)):
                    dice_score = dice_coefficient(outputs[i], masks[i], threshold=0.1)
                    train_dice_coefficients.append(dice_score)

                # Train the discriminator model
                if discriminator_model is not None:
                    discriminator_model.train()

                    discrim_labels = torch.tensor([1 if 'fake' in path else 0 for path in paths], dtype=torch.float32).to(device)
                    discriminator_optimizer.zero_grad()
                    discrim_preds = discriminator_model(images).squeeze(1)

                    loss_discrim = discriminator_criterion(discrim_preds, discrim_labels)
                    loss_discrim.backward()
                    discriminator_optimizer.step()

                    discrim_running_loss += loss_discrim.item() * images.size(0)

                    # 1 if fake (AI-generated)
                    # 0 if real
                    preds = (discrim_preds >= 0.5).long()
                    
                    discrim_correct += (preds == discrim_labels.long()).sum().item()
                    discrim_total += discrim_labels.size(0)

                    # Count correct predictions for fake and real samples
                    discrim_correct_fake += ((preds == 1) & (discrim_labels == 1)).sum().item()
                    discrim_total_fake += (discrim_labels == 1).sum().item()

                    discrim_correct_real += ((preds == 0) & (discrim_labels == 0)).sum().item()
                    discrim_total_real += (discrim_labels == 0).sum().item()


            # Average Dice across all training samples
            train_dice_mean = np.mean(train_dice_coefficients)
            train_dice_coefficient_history.append(train_dice_mean)

            # Save to CSV
            pd.Series(train_dice_coefficient_history).to_csv(os.path.join(exp_dir, 'train_dice_coefficients.csv'), index=False)

            # Average loss across all training samples
            train_loss = train_loss / len(train_loader.dataset)
            train_loss_history.append(train_loss)

            
            if train_loss < best_train_loss or epoch == 0:
                print('New best train loss:', train_loss)
                best_train_loss_dice_coefficients = dice_coefficients
                best_train_loss = train_loss

            pd.Series(train_loss_history).to_csv(os.path.join(exp_dir, 'train_loss_history.csv'), index=False)
            # Cross-Validation (validation) set
            if kfold_cv_enabled:
                model.eval()
            
                cv_val_loss = 0.0
                cv_val_dice_coefficients = []
                cv_val_data = []

                with torch.no_grad():
                    for batch_idx, batch in enumerate(cv_val_loader):
                        images, masks, paths = batch
                        images, masks = images.to(device), masks.to(device)
                        outputs = model(images)

                        # Loss calculation
                        loss = criterion(outputs, masks)
                        cv_val_loss += loss.item() * images.size(0)
                        
                        # Compute Dice coefficient per sample
                        for i in range(images.size(0)):
                            dice_score = dice_coefficient(outputs[i], masks[i], threshold=0.1)
                            cv_val_dice_coefficients.append(dice_score)
                            cv_val_data.append({
                                'filename': os.path.basename(paths[i]),
                                'dice_coefficient': dice_score,
                                'epoch': epoch,
                            })

                # Average Dice across all cross-validation samples
                cv_dice_mean = np.mean(cv_val_dice_coefficients)
                cv_dice_coefficient_history.append(cv_dice_mean)
                pd.Series(cv_dice_coefficient_history).to_csv(os.path.join(exp_dir, 'cv_dice_coefficients.csv'), index=False)

                # Average loss across all cross-validation samples
                cv_val_loss = cv_val_loss / len(cv_val_loader.dataset)
                cv_loss_history.append(cv_val_loss)
                pd.Series(cv_loss_history).to_csv(os.path.join(exp_dir, 'cv_loss_history.csv'), index=False)

                # Store Dice coefficient history from cv_val_data
                pd.DataFrame(cv_val_data).to_csv(os.path.join(exp_dir, 'cv_validation_per_subj_statistics.csv'), mode='a', header=not os.path.exists(os.path.join(exp_dir, 'validation_per_subj_statistics.csv')), index=False)

            # Validation (testing) set
            dice_coefficients = []
            if val_loader is not None:
                model.eval()

                val_loss = 0.0
                case_days = []
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
                            current_epoch_predictions.append({
                                'original_mask': masks[i].cpu().numpy()[0] * 255,
                                'original_image': images[i].cpu().numpy()[0],
                                'predicted_mask': predicted_mask,
                                'path': path
                            })

                        # Compute Dice coefficient per sample
                        for i in range(images.size(0)):
                            dice_score = dice_coefficient(outputs[i], masks[i], threshold=0.1)
                            dice_coefficients.append(dice_score)
                            case_day = paths[i].split('/')[-1].split('_slice')[0]
                            case_days.append(case_day)

                # Aggregate the per-caseday statistics
                # The statistics may already exist. We want to append to them if they do.
                pd.DataFrame({
                    'case_day': case_days,
                    'dice_coefficient': dice_coefficients,
                    'epoch': epoch 
                }).to_csv(os.path.join(exp_dir, 'validation_per_subj_statistics.csv'), mode='a', header=not os.path.exists(os.path.join(exp_dir, 'validation_per_subj_statistics.csv')), index=False)
                

            # Export discriminator statistics to CSV
            if discriminator_model is not None:
                # TODO: Remove validation code. not needed; there is no synthetic data in validation set
                discriminator_statistics = os.path.join(exp_dir, 'discriminator_statistics.csv')

                # Add all new statistics to the CSV file using pandas
                new_data = {
                    'epoch': epoch,

                    'discriminator_train_loss': discrim_running_loss / discrim_total if discrim_total > 0 else 0,
                    'discriminator_train_accuracy': discrim_correct / discrim_total if discrim_total > 0 else 0,
                    'discriminator_train_fake_accuracy': discrim_correct_fake / discrim_total_fake if discrim_total_fake > 0 else 0,
                    'discriminator_train_real_accuracy': discrim_correct_real / discrim_total_real if discrim_total_real > 0 else 0,
                    'discriminator_train_samples': discrim_total,
                }

                if not os.path.exists(discriminator_statistics):
                    new_data = pd.DataFrame([new_data])
                else:
                    # Load existing data, append new data, and save
                    existing_data = pd.read_csv(discriminator_statistics)
                    new_data = pd.DataFrame([new_data])
                    new_data = pd.concat([existing_data, new_data], ignore_index=True)
                new_data.to_csv(discriminator_statistics, index=False)

            # Average Dice across all validation samples
            dice_mean = np.mean(dice_coefficients)
            dice_std = np.std(dice_coefficients)
            val_dice_coef_history.append(dice_mean)
            val_dice_coef_history_std.append(dice_std)

            pd.Series(val_dice_coef_history).to_csv(os.path.join(exp_dir, 'val_dice_coefficients.csv'), index=False)

            # Average loss across all validation samples
            val_loss = val_loss / len(val_loader.dataset)
            val_loss_history.append(val_loss)
            pd.Series(val_loss_history).to_csv(os.path.join(exp_dir, 'val_loss_history.csv'), index=False)

            save_path = os.path.join(exp_dir, 'model', 'best_model.pt')
            torch.save(model.state_dict(), save_path)
            print(f"Final model saved to {save_path}")

            if val_loader is not None:
                print(f'Epoch {epoch+1}/{args.epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
                print(f"Dice Coefficient for Epoch {epoch}: {dice_mean:.4f}")
            else:
                print(f'Epoch {epoch+1}/{args.epoch}, Train Loss: {train_loss:.4f}, CV Val Loss: {cv_val_loss:.4f}')
                print(f"Cross-Validation Dice Coefficient for Epoch {epoch}: {cv_dice_mean:.4f}")
            if kfold_cv_enabled:
                print(f'Cross-Validation Loss: {cv_val_loss:.4f}, Cross-Validation Dice Coefficient: {cv_dice_mean:.4f}')
                # wandb.log({
                #     "epoch": epoch,
                #     'gen_model': args.gen_model,
                #     'synthetic_real_ratio': args.synthetic_real_ratio,
                #     "train_loss": train_loss,
                #     "val_loss": val_loss,
                #     "dice_coefficient": dice_mean,
                #     "predictions": [wandb.Image(pred[0], caption=os.path.basename(pred[1])) for pred in current_epoch_predictions]
                # })

            # Save the model and predictions only if Dice coefficient improves
            # if dice_mean > best_dice:
            #     best_dice = dice_mean
            #     best_dice_epoch = epoch
            #     best_dice_predictions = current_epoch_predictions  # Update best predictions

              
            #     patience_counter = 0
            # else:
            #     patience_counter += 1

        # Reset the model
        # model.train()
        # model.zero_grad()

            # if patience_counter >= patience_limit:
            #     print(f"Stopping early after {epoch + 1} epochs due to no improvement in Dice coefficient.")
            #     break

        # Save predictions for the best Dice epoch to disk
        
        # save_dir = os.path.join(exp_dir, 'predictions')
        # os.makedirs(save_dir, exist_ok=True)
        # for predicted_mask, original_path in best_dice_predictions:
        #     # Save the predicted mask with a consistent filename
        #     original_filename = os.path.basename(original_path).replace("_image.png", "_predicted.png")
        #     save_path = os.path.join(save_dir, original_filename)
        #     cv2.imwrite(save_path, predicted_mask)
        print(f"Predictions for the best Dice epoch ({best_dice_epoch + 1}) saved.")

        # Save best Dice epoch info to a text file
        best_dice_path = os.path.join(exp_dir, 'best_dice_epoch.txt')
        with open(best_dice_path, 'w') as f:
            f.write(f"Best Epoch: {best_dice_epoch + 1}\n")
            f.write(f"Dice Coefficient: {best_dice:.4f}\n")
            f.write(f"Best Train Loss: {min(train_loss_history):.4f}\n")
            # f.write(f"Best Val Loss: {min(val_loss_history):.4f}\n")
            # f.write(f"Best Val Loss: {min(val_loss_history):.4f}\n")
            f.write(f"Train Loss History: {json.dumps(train_loss_history)}\n")
            # f.write(f"Val Loss History: {json.dumps(val_loss_history)}\n")

        # Save the final epoch predictions to exp_dir
        for epoch_prediction in current_epoch_predictions:
            original_mask = epoch_prediction['original_mask']
            predicted_mask = epoch_prediction['predicted_mask']
            path = epoch_prediction['path']
            original_image = epoch_prediction['original_image']

            # Save the predicted mask with a consistent filename
            original_filename = os.path.basename(path).replace("-image.png", "-predicted.png")
            save_path = os.path.join(exp_dir, 'predictions', original_filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, predicted_mask)
            # Save the original image with a consistent filename
            original_image_filename = os.path.basename(path).replace("-image.png", "-original_image.png")
            original_image_save_path = os.path.join(exp_dir, 'predictions', original_image_filename)
            cv2.imwrite(original_image_save_path, original_image * 255)  # Assuming original_image is normalized

            # Save the original mask with a consistent filename
            original_mask_filename = os.path.basename(path).replace("-image.png", "-original_mask.png")
            original_mask_save_path = os.path.join(exp_dir, 'predictions', original_mask_filename)
            cv2.imwrite(original_mask_save_path, original_mask)
        print(f"Final epoch predictions saved to {os.path.join(exp_dir, 'predictions')}")

        ending_dice_info_path = os.path.join(exp_dir, 'ending_dice_info.json')
        # Write the file as a json file
        json_data = {
            'best_dice_epoch': best_dice_epoch + 1,
            'best_dice_coefficient': np.max(val_dice_coef_history),
            'best_train_loss': min(train_loss_history),
            'dice_coefficients': dice_coefficients,
            'best_train_loss_dice_coefficients': best_train_loss_dice_coefficients,
            'final_dice_coefficient': val_dice_coef_history[-1],
            'final_dice_coefficient_std': val_dice_coef_history_std[-1],
            'final_train_loss': train_loss_history[-1],
            'train_loss_history': train_loss_history,
            'train_dice_coefficient_history': train_dice_coefficient_history,
        }

        if kfold_cv_enabled:
            cv_data = {
                'best_cv_val_loss': min(cv_loss_history),
                'final_cv_val_loss': cv_loss_history[-1],
                'cv_val_loss_history': cv_loss_history,
                'cv_val_dice_coefficient_history': cv_dice_coefficient_history
            }
            # Add each to the json data
            json_data.update(cv_data)

        with open(ending_dice_info_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        
        # print(f"Best Dice epoch details saved to {best_dice_path}")
    
        # Plot and save the training, validation, and cross-validation loss curves: loss vs. epoch
        plot_save_path = os.path.join(exp_dir, 'loss_curve.jpg')
        plt.figure()
        plt.plot(train_loss_history, label='Train Loss')
        plt.plot(cv_loss_history, label='Cross-Validation Loss')
        if val_loader is not None:
            plt.plot(val_loss_history, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Over Epochs')
        plt.legend()
        os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
        plt.savefig(plot_save_path, format='jpg', bbox_inches='tight', pad_inches=0, dpi=100)

        # Plot and save the Dice coefficient curves
        plot_save_path = os.path.join(exp_dir, 'dice_coefficient_curve.jpg')
        plt.figure()
        plt.plot(train_dice_coefficient_history, label='Train Dice Coefficient')
        plt.plot(cv_dice_coefficient_history, label='Cross-Validation Dice Coefficient')
        if val_loader is not None:
            plt.plot(val_dice_coef_history, label='Validation Dice Coefficient')
        plt.xlabel('Epoch')
        plt.ylabel('Dice Coefficient')
        plt.title('Dice Coefficient Over Epochs')
        plt.legend()
        os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
        plt.savefig(plot_save_path, format='jpg', bbox_inches='tight', pad_inches=0, dpi=100)

        # Clear the figure
        plt.close()
        print(f"Loss and Dice coefficient plots saved to {plot_save_path}")


        # Plot and save Dice coefficient curve
        # plot_save_path = plot_metric(x=dice_coef_history,
        #             label="Dice Coefficient",
        #             plot_dir=exp_dir,
        #             args=args,
        #             metric='dice_coeff')

        # Save the curve to wandb
        # wandb.log({
        #     "dice_coefficient_curve": wandb.Image(plot_save_path),
        #     "synthetic_real_ratio": args.synthetic_real_ratio,
        #     "gen_model": args.gen_model
        # })

        print("Training completed.")

if __name__ == "__main__":
    args = train_arg_parser()
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize the datasets
    print(f"Loading training data from: {args.train_dir}")
    train_dataset = MadisonDatasetLabeled(args.train_dir, augment=True)
    print(f"Training dataset size: {len(train_dataset)}")

    val_loader = None
    if args.val_dir is not None:
        print(f"Loading validation data from: {args.val_dir}")
        val_dataset = MadisonDatasetLabeled(args.val_dir, augment=False)

        print(f"Validation dataset size: {len(val_dataset)}")
        val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False) 

    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)

    print('UNet architecture:', args.unet_architecture)

    if args.unet_architecture == 'uwm-unet':
        model = smp.Unet(
            encoder_name='efficientnet-b1',      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,        # model output channels (number of classes in your dataset)
            activation=None,
        )
        model = model.to(device)
    else:
        model = BaseUNet(in_channels=1, out_channels=1).to(device)

    print('Learning rate:', args.lr)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(30000/args.bs*args.epoch)+50, 
        eta_min=1e-6
    )

    discriminator_model = CNNClassifier().to(device) if args.gen_model != 'none' else None

    timestamp = np.datetime64('now', 's').astype(str).replace(':', '-')

    wandb.init(
        project="gsoc-diffusion-2025",
        name=f"unet_{args.gen_model}_benchmark_{timestamp}"
    )

    train_model(
        # model=model,
                train_dataset=train_dataset,
                val_loader=val_loader,
                # optimizer=optimizer,
                # scheduler=scheduler,
                # criterion=criterion,
                device=device,
                args=args,
                discriminator_model=None)

