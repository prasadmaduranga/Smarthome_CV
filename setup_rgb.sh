
# Generate data
# python  ./tools/data_gen/smarthome_gendata.py # generate skeelton data
# python  ./tools/data_gen/smarthome_rgb_gendata.py # generate rgb data

# Run : train spatio temporal model
#module load python3/3.9.2
#source /scratch/l12/pm9132/smarthome_env/bin/activate
#python main.py --config ./config/smarthome/cross_subject/train_joint.yaml


# Run : train rgb model
module load python3/3.9.2
source /scratch/l12/pm9132/smarthome_env/bin/activate
python main_rgb_joint.py --config ./config/smarthome/cross_subject/train_rgb_joint.yaml