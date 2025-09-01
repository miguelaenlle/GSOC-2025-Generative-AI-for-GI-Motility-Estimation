# Author: Miguel Aenlle

import os
import json
import subprocess
import numpy as np
import pandas as pd
import constants

def benchmark_unet(
    training_folder,
    finetuning_folder,
    validation_folder,
    gen_model = "",
    cv_subject = "",
    expansion_factor = 1,
    num_epochs = 30,
    device='cuda:0',
    unet_architecture='',
    lr=1e-4,
    eval_only=False,
    model_weights_path=None,
    self_supervised_finetuning=False,
    pixelwise_confidence_threshold=0.9
):
    timestamp = np.datetime64('now', 's').astype(str).replace(':', '-')
    exp_id = f'model-{gen_model}-expansion-factor-{expansion_factor}-cv-subject-{cv_subject}-{timestamp}'
    python_command = f"python '{constants.TRAIN_EVAL_UNET_PY}' --train_dir '{training_folder}' --val_dir '{validation_folder}' --finetuning_dir '{finetuning_folder}' --gen_model {gen_model} --exp_id '{exp_id}' --epoch {num_epochs} --device={device} --unet_architecture={unet_architecture} --lr={lr} --eval_only={eval_only} --model_weights_path={model_weights_path} --self_supervised_finetuning='{self_supervised_finetuning}' --pixelwise_confidence_threshold={pixelwise_confidence_threshold}"
    try:
        print(f"Running command: {python_command}")
        result = subprocess.run(python_command, shell=True, check=True)
        print("UNet training completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during UNet training: {e}")
    return exp_id

def main(
    models_to_run: list[str],
    eval_only: bool = False,
    device: str = 'cuda:0'
):
    os.chdir(os.path.abspath('../../1 - Segmentation/UNet-and-Synthesis-Results'))

    timestamp = np.datetime64('now', 's').astype(str).replace(':', '-')
    exp_ids_path = f'../exp_ids/{timestamp}'

    os.makedirs(exp_ids_path, exist_ok=True)

    exp_ids = []
    for model in models_to_run:
        model_info = constants.MODEL_INFO[model]
        best_model_paths = model_info.get('best_model_paths', {})
        dataset_path = model_info['dataset_path']

        for expansion_factor in model_info['expansion_factors']:
            datestamp = np.datetime64('now', 's').astype(str).replace(':', '-')

            exp_id = f'{model}-{datestamp}'
            experiment_logs_folder_path = f'../experiment-logs/{exp_id}'
            os.makedirs(experiment_logs_folder_path, exist_ok=True)

            all_data = []

            folder = f'augmented_dataset_expansion_factor_{expansion_factor}'


            cv_subjects = model_info.get('custom_cv_subjects', os.listdir(os.path.join(dataset_path, folder)))

            for cv_subject in cv_subjects:
                train_folder_path = os.path.join(dataset_path, folder, cv_subject, 'train')
                finetuning_folder_path = os.path.join(dataset_path, folder, cv_subject, 'finetuning_augmented')
                val_folder_path = os.path.join(dataset_path, folder, cv_subject, 'val')

                num_train_folder_non_mask_samples = len([f for f in os.listdir(train_folder_path) if 'mask' not in f])
                num_val_folder_non_mask_samples = len([f for f in os.listdir(val_folder_path) if 'mask' not in f])

                model_weights_path = best_model_paths.get(cv_subject, None) if best_model_paths else None
                if eval_only:
                    if model_weights_path:
                        print(f"Using model weights from {model_weights_path} for {cv_subject}")
                    else:
                        print(f'No model weights specified for {cv_subject}, skipping.')
                        continue

                exp_id = benchmark_unet(
                    training_folder=train_folder_path,
                    finetuning_folder=finetuning_folder_path,
                    validation_folder=val_folder_path,
                    gen_model=cv_subject,
                    cv_subject=cv_subject,
                    expansion_factor=expansion_factor,
                    num_epochs=30,
                    device=device,
                    unet_architecture=model_info['architecture'],
                    lr=2e-3,
                    eval_only=eval_only,
                    model_weights_path=model_weights_path if eval_only else None,
                    self_supervised_finetuning=model_info.get('self_supervised_finetuning', False),
                    pixelwise_confidence_threshold=model_info.get('pixelwise_confidence_threshold', 0.9)
                )
                # Obtain the results
                path = os.path.join(constants.UNET_LOGS_PATH, exp_id, 'fold_1', 'ending_dice_info.json')
                # Pull the JSON data
                json_data = json.load(open(path, 'r'))
                json_data['expansion_factor'] = expansion_factor
                json_data['cv_subject'] = cv_subject
                json_data['exp_id'] = exp_id
                json_data['num_train_samples'] = num_train_folder_non_mask_samples
                json_data['num_val_samples'] = num_val_folder_non_mask_samples

                all_data.append(json_data)

                filename = f'unet_performance_statistics_{cv_subject}_{expansion_factor}.csv'
                print('Filename:', filename)
                export_path = os.path.join(experiment_logs_folder_path, filename)
                pd.DataFrame(all_data).to_csv(export_path, index=False)
                print(f"UNet performance statistics saved to {export_path}")

                exp_ids.append({
                    'exp_id': exp_id,
                    'cv_subject': cv_subject,
                    'unet_performance_statistics_path': os.path.abspath(export_path),
                    'model_type': model,
                    'train_set_size': num_train_folder_non_mask_samples,
                    'val_set_size': num_val_folder_non_mask_samples,
                    'json_path': os.path.abspath(path)
                })

                pd.DataFrame(exp_ids).to_csv(f'{exp_ids_path}/exp_ids.csv', index=False)
    
if __name__ == "__main__":
    main(
        models_to_run=[
            'uwm-unet-singan-finetuned-n-5-10x', 
        ],
        eval_only = False,
        device = 'cuda:0'
    )