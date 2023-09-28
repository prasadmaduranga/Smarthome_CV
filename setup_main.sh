
# Generate data
# python  ./tools/data_gen/smarthome_gendata.py # generate skeelton data
# python  ./tools/data_gen/smarthome_rgb_gendata.py # generate rgb data

# Run : train spatio temporal model
module load python3/3.9.2
source /scratch/l12/pm9132/smarthome_env/bin/activate
python main.py --config ./config/smarthome/cross_subject/train_joint.yaml
python main.py --config ./config/nturgbd-cross-view/test_joint.yaml