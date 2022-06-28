import torch.utils.data as data

import os
import os.path
from os import listdir
from os.path import join
import numpy as np
from numpy.random import randint
import torch
import re

from colorama import init
from colorama import Fore, Back, Style
from torchvision import transforms
from PIL import Image

init(autoreset=True)


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r"(\d+)", str(text))]


class TSNDataSet(data.Dataset):
    def __init__(
        self,
        root_path,
        list_file,
        num_dataload,
        num_segments=3,
        new_length=1,
        modality="RGB",
        image_tmpl="img_{:05d}.t7",
        transform=None,
        force_grayscale=False,
        random_shift=True,
        test_mode=False,
        epic_kitchens=False,
        frame_size=224
    ):

        self.epic_kitchens = epic_kitchens
        self.frame_size = frame_size
        self.data_transform = transforms.Compose(
            [
                transforms.Resize(self.frame_size),
                transforms.CenterCrop(self.frame_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.num_dataload = num_dataload

        if (
            self.modality == "RGBDiff"
            or self.modality == "RGBDiff2"
            or self.modality == "RGBDiffplus"
        ):
            self.new_length += 1  # Diff needs one more image to calculate diff

        if self.epic_kitchens:

            self.videos_with_class = []
            self.ek_videos = {}

            with open(root_path, "r") as filelist:
                for line in filelist:
                    split_line = line.split()
                    path = split_line[0]
                    start_frame = int(split_line[1])
                    stop_frame = int(split_line[2])
                    label = int(split_line[3])
                    self.videos_with_class.append(
                        (path, start_frame, stop_frame, label)
                    )
                    kitchen = path.split("/")[-1]
                    if kitchen not in self.ek_videos:
                        kitchen_videos = self.find_frames(path)
                        kitchen_videos.sort(key=natural_keys)
                        self.ek_videos[kitchen] = kitchen_videos
        else:

            self._parse_list()  # read all the video files

    def _load_feature(self, directory, idx):
        if (
            self.modality == "RGB"
            or self.modality == "RGBDiff"
            or self.modality == "RGBDiff2"
            or self.modality == "RGBDiffplus"
        ):
            feat_path = os.path.join(directory, self.image_tmpl.format(idx))
            try:
                feat = [torch.load(feat_path)]
            except:
                print(Back.RED + feat_path)
            return feat

        elif self.modality == "Flow":
            x_feat = torch.load(
                os.path.join(directory, self.image_tmpl.format("x", idx))
            )
            y_feat = torch.load(
                os.path.join(directory, self.image_tmpl.format("y", idx))
            )

            return [x_feat, y_feat]

    def _parse_list(self):
        self.video_list = [
            VideoRecord(x.strip().split(" ")) for x in open(self.list_file)
        ]
        # repeat the list if the length is less than num_dataload (especially for target data)
        n_repeat = self.num_dataload // len(self.video_list)
        n_left = self.num_dataload % len(self.video_list)
        self.video_list = self.video_list * n_repeat + self.video_list[:n_left]

    # selects frames from input sequence
    def find_frames(self, video):
        frames = [join(video, f) for f in listdir(video) if self.is_img(f)]
        return frames

    # checks if input is image
    def is_img(self, f):
        return str(f).lower().endswith("jpg") or str(f).lower().endswith("jpeg")

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """
        # np.random.seed(1)

        if self.epic_kitchens:
            kitchen = record["video"].split("/")[-1]
            frame_paths = self.ek_videos[kitchen]
            frame_paths = frame_paths[record["start_frame"]:record["stop_frame"]]
            length = len(frame_paths)
        else:
            length = record.num_frames

        average_duration = (
            length - self.new_length + 1
        ) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(
                list(range(self.num_segments)), average_duration
            ) + randint(average_duration, size=self.num_segments)
        elif length > self.num_segments:
            offsets = np.sort(
                randint(length - self.new_length + 1, size=self.num_segments)
            )
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):

        if self.epic_kitchens:
            kitchen = record["video"].split("/")[-1]
            frame_paths = self.ek_videos[kitchen]
            frame_paths = frame_paths[record["start_frame"]:record["stop_frame"]]
            length = len(frame_paths)
        else:
            length = record.num_frames

        num_min = self.num_segments + self.new_length - 1
        num_select = length - self.new_length + 1

        if length >= num_min:
            tick = float(num_select) / float(self.num_segments)
            offsets = np.array(
                [int(tick / 2.0 + tick * float(x)) for x in range(self.num_segments)]
            )
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):

        if self.epic_kitchens:
            kitchen = record["video"].split("/")[-1]
            frame_paths = self.ek_videos[kitchen]
            frame_paths = frame_paths[record["start_frame"]:record["stop_frame"]]
            length = len(frame_paths)
        else:
            length = record.num_frames

        num_min = self.num_segments + self.new_length - 1
        num_select = length - self.new_length + 1

        if length >= num_min:
            tick = float(num_select) / float(self.num_segments)
            offsets = np.array(
                [int(tick / 2.0 + tick * float(x)) for x in range(self.num_segments)]
            )  # pick the central frame in each segment
        else:  # the video clip is too short --> duplicate the last frame
            id_select = np.array([x for x in range(num_select)])
            # expand to the length of self.num_segments with the last element
            id_expand = (
                np.ones(self.num_segments - num_select, dtype=int)
                * id_select[id_select[0] - 1]
            )
            offsets = np.append(id_select, id_expand)

        return offsets + 1

    def __getitem__(self, index):

        if self.epic_kitchens:
            video, start_frame, stop_frame, y = self.videos_with_class[index]
            record = {
                "video": video,
                "start_frame": start_frame,
                "stop_frame": stop_frame
            }
        else:
            record = self.video_list[index]
            y = None

        if not self.test_mode:
            segment_indices = (
                self._sample_indices(record)
                if self.random_shift
                else self._get_val_indices(record)
            )
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices, y)

    def get(self, record, indices, label):

        if self.epic_kitchens:
            kitchen = record["video"].split("/")[-1]
            frame_paths = self.ek_videos[kitchen]
            frame_paths = frame_paths[record["start_frame"]:record["stop_frame"]]
            data = []
            for i, seg_ind in enumerate(indices):
                frame_path = frame_paths[seg_ind - 1]
                frame_img = Image.open(frame_path)
                frame_feat = self.data_transform(frame_img)
                data.append(frame_feat)
            process_data = torch.stack(data)
        else:
            frames = list()
            for seg_ind in indices:
                p = int(seg_ind)
                for i in range(self.new_length):
                    seg_feats = self._load_feature(record.path, p)
                    frames.extend(seg_feats)

                    if p < record.num_frames:
                        p += 1

            # process_data = self.transform(frames)
            process_data = torch.stack(frames)
            label = record.label()

        return process_data, label

    def __len__(self):
        if self.epic_kitchens:
            return len(self.videos_with_class)
        return len(self.video_list)
