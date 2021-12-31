import torch.utils.data as data
import data_augmentation as dag
from PIL import Image
import os, numpy as np, cv2

def default_loader(path, input_size, aug = True):
    img=Image.open(path).convert('RGB')
    img = img.resize((input_size, input_size))
    img=np.array(img)
    h, w = img.shape[:2]
    if aug:
        if np.random.random() > 0.5:
            img = dag.flip(img)
        img = dag.rotate(img, np.random.randint(4))
        if np.random.random() > 0.75:
            img = dag.gamma(img)
        if np.random.random() > 0.75:
            img = dag.blur(img)
        if np.random.random() > 0.75:
            img = dag.hsv(img)
    image = Image.fromarray(img)
    return image

class Mydataset(data.Dataset):
    def __init__(self, data_root, csv_file = '', transform=None, target_transform=None, input_size = 224, is_training = True):
        with open(csv_file) as fid:
            csv_lines = fid.readlines()[1:]
            
        filename2label = {}
        for line in csv_lines:
            tmp = line.strip().split(',')
            if len(tmp) == 2:
                filename, label = tmp
                filename2label[filename] = int(label)
                
        self.train_list = [(os.path.join(data_root, filename), label) for filename in os.listdir(data_root) if filename in filename2label]
        self.transform = transform
        self.target_transform = target_transform
        self.loader = default_loader
        self.is_training = is_training
        self.input_size = input_size

    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = self.loader(path, self.input_size, aug = self.is_training)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return len(self.imgs)
