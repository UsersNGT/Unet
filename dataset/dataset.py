import torch 
from torch.utils.data import Dataset 
form PIL import Image
import numpy as np
import os 


class Mydataset(Dataset):
    def __int__(self, img_dir, label_dir, mode, transform):
        assert mode in ["train", "test"]
        self.img_dir = img_dir 
        self.label_dir = label_dir
        self.mode = mode
        self.transform = transform
        self.data_info = self.get_data_info(img_dir, label_dir, mode)
    

    @staticmethod
    def get_data_info(img_dir, label_dir, mode):
        if not os.path.exists(img_dir):
            print(f"img_dir not exist!!")
            return 
        img_list = os.listdir(img_dir)

        if mode == "train":
            if not os.path.exists(label_dir):
                print(f"label_dir not exist!!")
            label_list = os.listdir(label_dir)
            data_info = []
            for name in img_list:
                if name in label_list:
                    data_info.append(name)
        else:
            data_info = img_list
        return data_info

    def __len__(self):
        return len(self.data_info)
    

    def __getitem__(self, idx):
        name = self.data_info[idx]
        img_path = os.path.join(self.img_dir, name)
        label_path = os.path.join(self.label_dir, name)

        img = Image.open(img_path)
        img = self.transform(img)
        if self.mode == "train":
            mask = Image.open(label_path).convert("1")
            shape = img.shape
            mask_img = mask.resize((shape[1], shape[2]), resample=Image.NEAREST)
            mask = np.asarray(mask_img)
            mask = np.where(mask, 1.0, 0)
            mask = torch.from_numpy(mask)
            return {
                # 'image': torch.as_tensor(img.copy()).float().contiguous(),
                # 'mask': torch.as_tensor(mask.copy()).long().contiguous()
                'image': img,
                'mask': mask
            }
        
        return {"image":img}


