# UniAda: Universal Adaptive Multi-objective Adversarial Attack for End-to-End Autonomous Driving Systems
This repository contains experiments conducted in the paper 'UniAda: Universal Adaptive Multi-objective Adversarial Attack for End-to-End Autonomous Driving Systems', accepted for IEEE Transactions on Reliability journal, 2024, available at [download](https://github.com/UniAdaRepo/UniAda/blob/main/UniAda_Universal_Adaptive_Multiobjective_Adversarial_Attack_for_End-to-End_Autonomous_Driving_Systems_TR.pdf).

**Abstract**:  Adversarial attacks play a pivotal role in testing and improving the reliability of deep learning (DL) systems.
Existing literature has demonstrated that subtle perturbations
to the input can elicit erroneous outcomes, thereby substantially
compromising the security of DL systems. This has emerged
as a critical concern in the development of DL-based safety-
critical systems like Autonomous Driving Systems (ADSs). The
focus of existing adversarial attack methods on End-to-End (E2E)
ADSs has predominantly centered on misbehaviors of steering
angle, which overlooks speed-related controls or imperceptible perturbations. To address these challenges, we introduce
UniAda—a multi-objective white-box attack technique with a
core function that revolves around crafting an image-agnostic
adversarial perturbation capable of simultaneously influencing
both steering and speed controls. UniAda capitalizes on an
intricately designed multi-objective optimization function with
the Adaptive Weighting Scheme (AWS), enabling the concurrent
optimization of diverse objectives. Validated with both simu-
lated and real-world driving data, UniAda outperforms five
benchmarks across two metrics, inducing steering and speed
deviations with 3.54◦ ∼ 29◦ and 11.0km/h ∼ 22.0km/h. This
systematic approach establishes UniAda as a proven technique
for adversarial attacks on modern DL-based E2E ADSs.

## Environment:
To successfully run the code, we provide a conda environment.yml file. Start by cloning the repository on your local folder and then, to install, just run:
<code>conda env create -f environment.yml</code>

## Path Setting:
Please set the root path as follows:
<code>UniAda_root=/path/to/clone/UniAda</code>
<code>cd $UniAda_root/</code>

## Dataset:
14 driving videos (both real-world and simulated) are provided in the folder -- data

## Download Pretrained models:
To download CILRS and CILR models, run:
<code>python3 ./tools/download_nocrash_models.py</code>

Download MotionTransformer from [download](https://onedrive.live.com/?authkey=%21AGHA3pWwTPDxkJg&id=98408C909B12E88E%213090&cid=98408C909B12E88E&parId=root&parQt=sharedby&o=OneUp)

## Train on Carla100 dataset:
Set the dataset path:\
<code>export COIL_DATASET_PATH=$UniAda_root/</code>

To run UniAda (modify args.title for different driving videos),\
<code>python3 update.py --method 'uni-dynamic' --title='episode_00000_pedestrain'</code>

To run baseline Perturbation Attack (all videos):\
cd Perturb_Att\
<code>python3 run_CILR.py</code>
<code>python3 run_CILRS.py</code>

To run baseline DeepManeuver (all videos):\
cd DM\
<code>python3 run.py</code>
<code>python3 run_CILRS.py</code>

To run baseline DeepBillboard (modify args.title for different driving videos):\
<code>python3 update.py --method 'deepbillboard' --title='episode_00000_pedestrian'</code>

To run baseline FGSM (modify args.title for different driving videos):\
<code>python3 update.py --method 'FGSM' --weights='equal' --iters 2 --title='episode_00000_pedestrian'</code>

To run baseline UniEqual (modify args.title for different driving videos):\
<code>python3 update.py --method 'uni-const' --weights='equal' --title='episode_00000_pedestrian'</code>


Please feel free to modify hyperparameters based on your need.

## Train on real-world dataset:
<code>cd MT</code>

To run UniAda (modify args.title for different driving videos):\
<code>python3 update.py --method 'uni-dynamic' --title='digital_Udacity_straight1'</code>

To run baseline Perturbation Attack (modify args.title for different driving videos):\
<code>python3 PA.py  --title='digital_Udacity_straight1'</code>

To run baseline DeepManeuver (modify args.title for different driving videos):\
<code>python3 DM.py  --title='digital_Udacity_straight1'</code>

To run baseline DeepBillboard (modify args.title for different driving videos):\
<code>python3 update.py --method 'deepbillboard' --title='digital_Udacity_straight1'</code>

To run baseline FGSM (modify args.title for different driving videos):\
<code>python3 update.py --method 'FGSM' --weights='equal' --iters 2 --title='digital_Udacity_straight1'</code>

To run baseline UniEqual (modify args.title for different driving videos):\
<code>python3 update.py --method 'uni-const' --weights='equal' --title='digital_Udacity_straight1'</code>

