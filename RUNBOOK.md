# DA6401 Assignment 2 Runbook

This file tells you exactly what to do, where to keep data and checkpoints, how to train each task, and how to generate the W&B report assets.

## 1. Paths

Project root on your machine:

```text
C:\Users\yagni\OneDrive\Desktop\IITM Course Notes\Intro to DL DA6401\da6401_assignment_2
```

Recommended dataset path:

```text
C:\Users\yagni\OneDrive\Desktop\IITM Course Notes\Intro to DL DA6401\da6401_assignment_2\datasets\oxford-iiit-pet
```

Recommended checkpoint path:

```text
C:\Users\yagni\OneDrive\Desktop\IITM Course Notes\Intro to DL DA6401\da6401_assignment_2\checkpoints
```

Expected checkpoint filenames:

- `checkpoints\classifier.pth`
- `checkpoints\localizer.pth`
- `checkpoints\unet.pth`
- `checkpoints\multitask.pth`

Recommended output path:

```text
C:\Users\yagni\OneDrive\Desktop\IITM Course Notes\Intro to DL DA6401\da6401_assignment_2\outputs
```

## 2. What Is Implemented

- Task 1: VGG11 classifier from scratch with BatchNorm and custom dropout
- Task 2: localization head on top of the VGG11 encoder
- Task 3: VGG11 encoder + U-Net style decoder with transposed convolutions
- Task 4: unified multi-task model that loads the three task checkpoints
- root `multitask.py` wrapper for `from multitask import MultiTaskPerceptionModel`
- autograder-facing classes:
  - `models.vgg11.VGG11`
  - `models.layers.CustomDropout`
  - `losses.iou_loss.IoULoss`
  - `multitask.MultiTaskPerceptionModel`

## 3. Setup

Open PowerShell in the assignment folder:

```powershell
cd "C:\Users\yagni\OneDrive\Desktop\IITM Course Notes\Intro to DL DA6401\da6401_assignment_2"
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Notes:

- `wandb` is optional unless you want the report logs.
- `albumentations` is listed in requirements, but the dataset loader also has a fallback path if it is unavailable.

## 4. W&B Setup

Login once:

```powershell
wandb login
```

Optional environment variables:

```powershell
setx WANDB_PROJECT "da6401_assignment_2"
setx WANDB_ENTITY "<your_wandb_username_or_team>"
```

Recommended runtime flags:

- `--wandb_mode online` for report runs
- `--wandb_mode offline` for local logging that you will sync later
- `--wandb_mode disabled` while debugging

## 5. Dataset Download

Download and extract Oxford-IIIT Pet:

```powershell
python train.py --download_data --prepare_data_only --data_root "datasets/oxford-iiit-pet"
```

Expected structure:

```text
datasets\oxford-iiit-pet\images
datasets\oxford-iiit-pet\annotations
datasets\oxford-iiit-pet\annotations\trimaps
datasets\oxford-iiit-pet\annotations\xmls
```

## 6. Model Assumptions

- fixed input size: `224 x 224`
- normalized inputs
- localization output format: `[x_center, y_center, width, height]`
- localization targets are in resized image pixel coordinates
- localization training loss: `MSE + IoULoss`
- segmentation training loss: `CrossEntropy + DiceLoss`
- segmentation classes: 3-class trimap

## 7. Train Each Task

### Task 1: Classification

```powershell
python train.py --task classification --data_root "datasets/oxford-iiit-pet" --epochs 25 --batch_size 16 --learning_rate 1e-3 --dropout_p 0.5 --use_batch_norm true --wandb_mode online --wandb_run_name "task1_classifier"
```

Output:

```text
checkpoints\classifier.pth
```

### Task 2: Localization

```powershell
python train.py --task localization --data_root "datasets/oxford-iiit-pet" --epochs 20 --batch_size 16 --learning_rate 1e-4 --freeze_strategy partial --load_encoder_from_classifier true --classifier_checkpoint "checkpoints/classifier.pth" --wandb_mode online --wandb_run_name "task2_localizer"
```

Output:

```text
checkpoints\localizer.pth
```

### Task 3: Segmentation

```powershell
python train.py --task segmentation --data_root "datasets/oxford-iiit-pet" --epochs 25 --batch_size 8 --learning_rate 1e-4 --freeze_strategy none --load_encoder_from_classifier true --classifier_checkpoint "checkpoints/classifier.pth" --wandb_mode online --wandb_run_name "task3_unet"
```

Output:

```text
checkpoints\unet.pth
```

### Task 4: Unified Multi-Task Model

```powershell
python train.py --task multitask --data_root "datasets/oxford-iiit-pet" --epochs 15 --batch_size 8 --learning_rate 1e-4 --classifier_checkpoint "checkpoints/classifier.pth" --localizer_checkpoint "checkpoints/localizer.pth" --unet_checkpoint "checkpoints/unet.pth" --wandb_mode online --wandb_run_name "task4_multitask"
```

Output:

```text
checkpoints\multitask.pth
```

## 8. Evaluate

Classification:

```powershell
python inference.py --task classification --mode dataset --split val --data_root "datasets/oxford-iiit-pet" --classifier_checkpoint "checkpoints/classifier.pth"
```

Localization:

```powershell
python inference.py --task localization --mode dataset --split val --data_root "datasets/oxford-iiit-pet" --localizer_checkpoint "checkpoints/localizer.pth"
```

Segmentation:

```powershell
python inference.py --task segmentation --mode dataset --split val --data_root "datasets/oxford-iiit-pet" --unet_checkpoint "checkpoints/unet.pth"
```

Multi-task:

```powershell
python inference.py --task multitask --mode dataset --split val --data_root "datasets/oxford-iiit-pet" --classifier_checkpoint "checkpoints/classifier.pth" --localizer_checkpoint "checkpoints/localizer.pth" --unet_checkpoint "checkpoints/unet.pth"
```

## 9. Run On New Images

Single image:

```powershell
python inference.py --task multitask --mode image --input "custom_images\pet1.jpg" --output_dir "outputs\single_demo" --classifier_checkpoint "checkpoints/classifier.pth" --localizer_checkpoint "checkpoints/localizer.pth" --unet_checkpoint "checkpoints/unet.pth"
```

Folder of images:

```powershell
python inference.py --task multitask --mode showcase --input_dir "custom_images" --output_dir "outputs\showcase" --classifier_checkpoint "checkpoints/classifier.pth" --localizer_checkpoint "checkpoints/localizer.pth" --unet_checkpoint "checkpoints/unet.pth"
```

## 10. Commands For Each W&B Report Question

### 2.1 BatchNorm effect

With BN:

```powershell
python train.py --task classification --use_batch_norm true --dropout_p 0.5 --epochs 25 --learning_rate 1e-3 --wandb_mode online --wandb_run_name "q2_1_bn_on"
```

Without BN:

```powershell
python train.py --task classification --use_batch_norm false --dropout_p 0.5 --epochs 25 --learning_rate 1e-3 --wandb_mode online --wandb_run_name "q2_1_bn_off"
```

Activation histogram for the 3rd conv layer:

```powershell
python inference.py --task classification --mode activation_hist --input "datasets\oxford-iiit-pet\images\Abyssinian_1.jpg" --checkpoint "checkpoints\classifier.pth" --output_dir "outputs\q2_1"
```

### 2.2 Dropout comparison

```powershell
python train.py --task classification --dropout_p 0.0 --wandb_mode online --wandb_run_name "q2_2_dropout_0"
python train.py --task classification --dropout_p 0.2 --wandb_mode online --wandb_run_name "q2_2_dropout_02"
python train.py --task classification --dropout_p 0.5 --wandb_mode online --wandb_run_name "q2_2_dropout_05"
```

### 2.3 Transfer learning showdown

Strict feature extractor:

```powershell
python train.py --task segmentation --freeze_strategy strict --wandb_mode online --wandb_run_name "q2_3_strict"
```

Partial fine-tuning:

```powershell
python train.py --task segmentation --freeze_strategy partial --wandb_mode online --wandb_run_name "q2_3_partial"
```

Full fine-tuning:

```powershell
python train.py --task segmentation --freeze_strategy none --wandb_mode online --wandb_run_name "q2_3_full"
```

### 2.4 Feature maps

```powershell
python inference.py --task classification --mode feature_maps --input "datasets\oxford-iiit-pet\images\beagle_1.jpg" --checkpoint "checkpoints\classifier.pth" --output_dir "outputs\q2_4"
```

### 2.5 Bounding box table

Recommended with the multi-task model so confidence comes from the classifier branch:

```powershell
python inference.py --task multitask --mode bbox_table --split val --num_samples 10 --data_root "datasets/oxford-iiit-pet" --output_dir "outputs\q2_5" --wandb_mode online --wandb_run_name "q2_5_bbox_table"
```

### 2.6 Dice vs pixel accuracy

Metrics:

```powershell
python inference.py --task segmentation --mode dataset --split val --data_root "datasets/oxford-iiit-pet" --unet_checkpoint "checkpoints\unet.pth"
```

Sample masks:

```powershell
python inference.py --task segmentation --mode mask_gallery --split val --num_samples 5 --data_root "datasets/oxford-iiit-pet" --output_dir "outputs\q2_6" --wandb_mode online --wandb_run_name "q2_6_masks"
```

### 2.7 Final pipeline showcase

```powershell
python inference.py --task multitask --mode showcase --input_dir "custom_images" --output_dir "outputs\q2_7_showcase" --wandb_mode online --wandb_run_name "q2_7_showcase"
```

### 2.8 Meta-analysis

Use the W&B runs from the commands above. The training script logs:

- total loss
- classification loss, accuracy, macro F1
- localization MSE, IoU loss, IoU
- segmentation CE, Dice loss, Dice score, pixel accuracy

## 11. Important Submission Notes

- checkpoint names already match the assignment
- `.gitignore` excludes dataset files, outputs, wandb logs, and `.pth` checkpoints
- the current code loads checkpoints locally with relative paths

Note about `gdown`:

- the assignment PDF restricts libraries tightly
- this code therefore does not depend on `gdown`
- if the course staff explicitly asks for Drive-download code inside `MultiTaskPerceptionModel`, add it only after confirming that `gdown` is allowed during grading

## 12. Shortest Working Command Order

```powershell
python train.py --download_data --prepare_data_only
python train.py --task classification --epochs 25 --batch_size 16 --learning_rate 1e-3 --wandb_mode online
python train.py --task localization --epochs 20 --batch_size 16 --learning_rate 1e-4 --freeze_strategy partial --classifier_checkpoint "checkpoints\classifier.pth" --wandb_mode online
python train.py --task segmentation --epochs 25 --batch_size 8 --learning_rate 1e-4 --classifier_checkpoint "checkpoints\classifier.pth" --wandb_mode online
python inference.py --task multitask --mode dataset --split val --classifier_checkpoint "checkpoints\classifier.pth" --localizer_checkpoint "checkpoints\localizer.pth" --unet_checkpoint "checkpoints\unet.pth"
```
