from saifooler.classifiers.classifier import Classifier

import torch
import json
import os


class ImageNetClassifier(Classifier):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)

        class_index_path = os.path.join(
            os.path.dirname(__file__),
            "imagenet_class_index.json"
        )
        self.class_dict = {
            int(key): val[1]
            for key, val in json.load(open(class_index_path)).items()
        }

    def to(self, device):
        super().to(device)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

    def get_class_label(self, class_id: int):
        return self.class_dict[class_id]

    def normalize_image(self, image):
        image = (image - self.mean) / self.std
        image = image.permute(2, 0, 1)
        image = image.unsqueeze(0)
        return image

