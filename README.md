# UniAda
This repository contains experiments conducted in the paper 'UniAda: Universal Adaptive Multi-objective Adversarial Attack for End-to-End Autonomous Driving Systems'

## Environment:
To successfully run the code, we provide a conda environment.yml file. Start by cloning the repository on your local folder and then, to install, just run:
<code>conda env create -f environment.yml</code>

## Path Setting:
<code>UniAda_root=/path/to/clone/UniAda</code>
<code> cd $UniAda_root/</code>

## Dataset:
14 driving videos are stored in the folder dataset

## Train on Carla100 dataset:
Set the dataset path:\
<code>export COIL_DATASET_PATH=$UniAda_root/</code>

To run UniAda,\
<code>python3 update.py --method 'uni-dynamic' --title='episode_00000_pedestrain'</code>

To run baseline Perturbation Attack (all videos):\
cd Perturb_Att\
<code>python3 run_CILR.py</code>

To run baseline DeepManeuver (all videos):\
cd DM\
<code>python3 run.py</code>

To run baseline DeepBillboard:\
<code>python3 update.py --method 'deepbillboard' --title='episode_00000_pedestrian'</code>

To run baseline FGSM:\
<code>python3 update.py --method 'FGSM' --weights='equal' --iters 2 --title='episode_00000_pedestrian'</code>

To run baseline UniEqual:\
<code>python3 update.py --method 'uni-const' --weights='equal' --title='episode_00000_pedestrian'</code>


Please feel free to modify hyperparameters based on your need.

## Train on real-world dataset:
cd MT\

To run UniAda,\
<code>python3 update.py --method 'uni-dynamic' --title='episode_00000_pedestrain'</code>

To run baseline Perturbation Attack,\
<code>python3 PA.py  --title='digital_Udacity_straight1'</code>

To run baseline DeepManeuver,\
<code>python3 DM.py  --title='digital_Udacity_straight1'</code>

To run baseline DeepBillboard,\
<code>python3 update.py --method 'deepbillboard' --title='digital_Udacity_straight1'</code>

To run baseline FGSM,\
<code>python3 update.py --method 'FGSM' --weights='equal' --iters 2 --title='digital_Udacity_straight1'</code>

To run baseline UniEqual,\
<code>python3 update.py --method 'uni-const' --weights='equal' --title='digital_Udacity_straight1'</code>
