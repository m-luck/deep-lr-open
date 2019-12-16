import matplotlib.pyplot as plt
import torch

from utils import zones

file_path = zones.get_model_latest_file_path(self.base_dir)
if file_path is None or not os.path.isfile(file_path):
    raise ValueError("{} not found".format(file_path))

checkpoint = torch.load(file_path)
print(checkpoint['train_epoch_losses'])