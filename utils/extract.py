from os import makedirs, listdir, system
from os.path import join, exists
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data as data
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.models import resnet18, resnet50, resnet101
from argparse import ArgumentParser
import torch
from tqdm import tqdm


parser = ArgumentParser()
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()


class ResNet(nn.Module):
    def __init__(self, pretrained=True, version="resnet101"):
        super().__init__()

        assert version in {"resnet18", "resnet50", "resnet101"}

        self.pretrained = pretrained
        self.version = version

        if self.version == "resnet18":
            self.model = nn.Sequential(
                *list(resnet18(pretrained=pretrained).children())[:-1]
            )
        elif self.version == "resnet50":
            self.model = nn.Sequential(
                *list(resnet50(pretrained=pretrained).children())[:-1]
            )
        else:
            self.model = nn.Sequential(
                *list(resnet101(pretrained=pretrained).children())[:-1]
            )

    def forward(self, X):
        features = self.model(X)
        return features


class MyDataset(data.Dataset):
    def __init__(self, path, frame_size=224):

        self.my_list = []
        self.frame_size = frame_size

        kitchens = ["P02", "P04"]
        for kitchen in kitchens:
            kitchen_path = join(path, kitchen, "rgb_frames")
            sub_folders = [d for d in listdir(kitchen_path) if not d.endswith(".tar")]
            for sub_folder in sub_folders:
                sub_folder_path = join(kitchen_path, sub_folder)
                frames = listdir(sub_folder_path)
                for frame in frames:
                    frame_path = join(sub_folder_path, frame)
                    obj = {
                        "path": frame_path,
                        "kitchen": kitchen,
                        "sub_folder": sub_folder,
                        "frame_name": frame.split(".")[0]
                    }
                    self.my_list.append(obj)

    def __len__(self):
        return len(self.my_list)

    def __getitem__(self, index):

        obj = self.my_list[index]
        frame = Image.open(obj["path"]).convert("RGB")
        frame = TF.resize(frame, self.frame_size)
        frame = TF.to_tensor(frame)
        return frame, obj["kitchen"], obj["sub_folder"], obj["frame_name"]


if exists(args.output):
    system("rm -r {}".format(args.output))
makedirs(args.output)

img_size = 224
batch_size = 1

dataset = MyDataset(args.path)

# create a data loader
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,
    shuffle=True,
    drop_last=False,
)

model = ResNet()

with torch.no_grad():
    for data in tqdm(dataloader):
        tensor = data[0]
        feat = model(tensor).view(batch_size, -1)
        kitchen = data[1][0]
        sub_folder = data[2][0]
        frame_name = data[3][0]
        output_kitchen = join(args.output, kitchen)
        if not exists(output_kitchen):
            makedirs(output_kitchen)
        output_sub_folder = join(output_kitchen, sub_folder)
        if not exists(output_sub_folder):
            makedirs(output_sub_folder)
        output_path = join(output_sub_folder, "{}.pt".format(frame_name))
        torch.save(feat, output_path)

