import argparse
import pickle
from tqdm import tqdm
import sys
import json
import numpy as np
import os
sys.path.extend(['../'])
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import cv2


# from data_gen.preprocess import pre_normalization

training_subjects = [3,4,6,7,9,12,13,15,17,19,25]
action_classes={
'Cook.Cleandishes':0,'Cook.Cleanup':1,'Cook.Cut':2, 'Cook.Stir':3, 'Cook.Usestove':4, 
'Cutbread':5, 'Drink.Frombottle':6, 'Drink.Fromcan':7, 'Drink.Fromcup':8, 'Drink.Fromglass':9,
'Eat.Attable':10, 'Eat.Snack':11, 'Enter':12,'Getup':13, 'Laydown':14, 'Leave':15, 
'Makecoffee.Pourgrains':16, 'Makecoffee.Pourwater':17, 'Maketea.Boilwater':18, 'Maketea.Insertteabag':19, 'Pour.Frombottle':20, 
'Pour.Fromcan':21, 'Pour.Fromkettle':22, 'Readbook':23, 'Sitdown':24, 'Takepills':25, 
'Uselaptop':26, 'Usetelephone':27, 'Usetablet':28, 'Walk':29, 'WatchTV':30
}

action_cv={
'Cutbread':0, 'Drink.Frombottle':1, 'Drink.Fromcan':2, 'Drink.Fromcup':3, 'Drink.Fromglass':4, 'Eat.Attable':5, 
'Eat.Snack':6, 'Enter':7, 'Getup':8, 'Leave':9, 'Pour.Frombottle':10, 'Pour.Fromcan':11, 
'Readbook':12, 'Sitdown':13, 'Takepills':14, 'Uselaptop':15, 'Usetablet':16,
'Usetelephone':17, 'Walk':18
}
 
training_cameras1 = [1]
training_cameras2 = [1,3,4,6,7]
val_cameras = [5]
testing_cameras=[2]

max_body_true = 1
max_body=2
num_joint = 15
max_frame = 4000
frame_rate = 25
resnet_encoder_dims = 1000




class ImageEncoder:
    def __init__(self, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = resnet50(pretrained=True).to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.batch_size = batch_size

    def encode_images(self, frames):
        imgs = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
        tensors = [self.transform(img) for img in imgs]
        batch_tensor = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            encodings = self.model(batch_tensor)
        return encodings.cpu().numpy()

encoder = ImageEncoder()

def read_xyz(file):
    cap = cv2.VideoCapture(file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = int(cap.get(cv2.CAP_PROP_FPS) / frame_rate)
    video_encodings = np.zeros((frame_count, resnet_encoder_dims))

    frames = []
    indices = []
    for i in range(0, frame_count, max(1, step)):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            indices.append(i)
            if len(frames) == encoder.batch_size:
                encodings = encoder.encode_images(frames)
                for j, index in enumerate(indices):
                    video_encodings[index, :] = encodings[j]
                frames, indices = [], []

    if frames:
        encodings = encoder.encode_images(frames)
        for j, index in enumerate(indices):
            video_encodings[index, :] = encodings[j]

    cap.release()
    return video_encodings

def gendata(data_path, out_path, ignored_sample_path=None, benchmark='xview', part='eval'):
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.json' for line in f.readlines()
            ]
    else:
        ignored_samples = []
    sample_name = []
    sample_label = []

    for filename in os.listdir(data_path):
        # if start with . continue
        if filename.startswith('.'):
            continue
        if (filename in ignored_samples) or ((filename.split('_')[0] not in action_cv.keys()) and (benchmark == 'xview1' or benchmark == 'xview2')) :
            continue
        if benchmark == 'xview1' or benchmark == 'xview2':
            action_class = int(
                action_cv[filename.split('_')[0]])
        else:
            action_class = int(action_classes[filename.split('_')[0]])
        subject_id = int(filename.split('_')[1][1:])
        camera_id = int(
            filename.split('_')[4][1:3])

        if benchmark == 'xview1':
            istraining = (camera_id in training_cameras1)
            istesting=(camera_id in testing_cameras)
            isval=(camera_id in val_cameras)
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        elif benchmark == 'xview2':
            istraining = (subject_id in training_cameras2)
            istesting=(camera_id in testing_cameras)
            isval=(camera_id in val_cameras)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'val':
            if benchmark == 'xview1' or  benchmark == 'xview2':
                issample = isval
            else:
                issample = not (istraining)
        elif part == 'test':
            if benchmark == 'xview1' or  benchmark == 'xview2':
                issample = istesting
            else:
                issample = not (istraining)
        else:
            raise ValueError()

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class)

    with open('{}/{}_rgb_joint_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    fp = np.zeros((len(sample_label),max_frame ,resnet_encoder_dims ), dtype=np.float32)

    for i, s in enumerate(tqdm(sample_name)):
        data = read_xyz(os.path.join(data_path, s))
        fp[i,0:data.shape[0],:] = data

    np.save('{}/{}_data_rgb_joint.npy'.format(out_path, part), fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Smarthome Data Converter.')
    # data/smarthome/smarthome_raw/trimmed_video_skeleton
    # parser.add_argument('--data_path', default='./data/smarthome/smarthome_raw/trimmed_video_skeleton/')

    parser.add_argument('--data_path', default='./data/smarthome/smarthome_raw/test_trimmed_video/')

    parser.add_argument('--ignored_sample_path',
                        default=None)
    parser.add_argument('--out_folder', default='./data/smarthome/')

    benchmark = ['xview1', 'xview2']
    part = ['train', 'val']# 'test']
    arg = parser.parse_args()
    print('raw_path: ', arg.data_path)
    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            print(b, p)

            gendata(
                arg.data_path,
                out_path,
                arg.ignored_sample_path,
                benchmark=b,
                part=p)

