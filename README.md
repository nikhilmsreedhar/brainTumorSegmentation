# brainTumorSegmentation
Deep Learning-based Brain Tumor Segmentation Using MRI scans.

Dataset used: 2016-2017 Brain Tumor Segmentation challenges (BraTS 2016 and 2017). *Disclaimer*: The Dataset used is extremely large and cannot be dsiplayed in this repository. To download the dataset, select the Task01_BrainTumour.tar file from the link as follows: https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2

Evaluation: The sub-regions considered for evaluation are: 1) the "enhancing tumor" (ET), 2) the "tumor core" (TC), and 3) the "whole tumor" (WT)
The classes or labels used for the purposes of this segmentation task are as follows: Background gray brain matter (label 0), necrotic (fluid-filled) and non-enhancing   tumor (label 1), peritumoral edema (label 2) and Gadolinium-enhancing tumor (label 4)
The segmentation accuracy is measured by the Dice score and the Hausdorff distance (95%) metrics for enhancing tumor region (ET, label 4), regions of the tumor core     (TC, labels 1 and 4), and the whole tumor region (WT, labels 1,2 and 4).

To execute the segmentation, first, the user must set up an anaconda/python environment with all relevant packages and requirements.

Next, the user should run the preprocess.py file in the data directory, where the Brain Tumor MRI .nii files are contained in another sub-directory named data under the data directory that preprocess.py is located in. 

Next, the user should run split.py to perform a train-test split to obtain a training set and a validation set. Since there were no labels provided in the test set, no test set was used.

Finally, to perform the training, the user should run train.py file. Using the command line, go to the directory in which train.py is stored in and execute the command 'python -m torch.distributed.launch --nproc_per_node=2 --master_port 20003 train.py --batch_size 3 --end_epoch 400'. This will execute the training process for 400 epochs with a batch size of 3.

If the user is utilizing a server with access to a GPU cluster (which is required for this project to run), simply go to the directory in which train.py is stored in and execute the command 'sbatch brainSeg.slurm'. To check the progress of the training process, execute the command 'cat tumor_seg.<Your Slurm JOB ID>.err'. This should output the Dice scores for the appropriate tumor regions.

This project required the use of two Tesla V100 GPUs to execute training.
