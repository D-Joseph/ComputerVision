import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image


class CustomDataLoader(Dataset):
    def __init__(self, )