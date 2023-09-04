Dataset : 
https://project.inria.fr/toyotasmarthome/

Skeleton analysis:
https://github.com/YangDi666/2s-AGCN-For-Daily-Living
- files referred from this one

RGB image analysis:
https://github.com/srijandas07/i3d_smarthome/tree/main

MMNet 
https://github.com/bruceyo/MMNet
- overral refeerence for the structure

Filenaming convention
Cook.Cleandishes_p02_r00_v02_c03.mp4
<action_class><subject_id><repeat_id><View_id><camera_id>


AGCN Master
https://github.com/lshiwjx/2s-AGCN

31 action classes
18 subjects


===================
-- 2s-AGCN GNNbased skeleton analysis - Steps for training and testing :

data gen step 
python  ./tools/data_gen/smarthome_gendata.py
// will create files in data folder

if needed 
pip uninstall torch numpy scipy
pip install torch numpy scipy

training 
python main_joint.py recognition -c config/<dataset>/train_joint.yaml [--work_dir <work folder>]
python main.py --config ./config/smarthome/cross_subject/train_joint.yaml

testing
python main.py --config ./config/nturgbd-cross-view/test_joint.yaml
python ensemble.py --datasets ntu/xview
// not tested yet


Data gen
python  ./tools/data_gen/smarthome_gendata.py          // for skeleton
pip freeze > requirements.txt      // for rgb resnet encoding

Model Training
python main_rgb_joint.py --config ./config/smarthome/cross_subject/train_rgb_joint.yaml

