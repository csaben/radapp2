from config import *
from dataset import *
from torch.utils.data import DataLoader
import torch
import keras

# transform = transforms.Compose([
#     transforms.Resize((240,240)),
#     transforms.ToTensor()
# ])
transform=None

tumor_dataset = ImageDataset(TUMOR_IMG, transform=transform, label=1)
control_dataset = ImageDataset(CONTROL_IMG, transform=transform, label=0)

dataset = torch.utils.data.ConcatDataset([tumor_dataset, control_dataset])


"""
TODO: Running into a sizing error for the model, I should add a preprocessing fn
      and just use the model input size as a hard rule rather than trying to make it
      work in the collate fn.

      also train a new torch one 
"""
def collate_fn(batch):
    max_size = tuple(max(s) for s in zip(*[img.shape for img, label in batch]))
    batch_size = len(batch)
    images = torch.zeros((batch_size,) + max_size, dtype=torch.float32)
    labels = torch.zeros(batch_size, dtype=torch.long)
    for i, (img, label) in enumerate(batch):
        img = img.resize((3, 240, 240))
        images[i] = img
        labels[i] = label
    return images, labels

dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

from keras.models import load_model

model = keras.models.load_model("../input/MODELS/MODELS/model_v1.h5")
# print(model.input_shape)
# (None, 240, 240, 3)

for inputs, labels in dataloader:
    # Apply the model to the inputs and labels
    print(labels)
    outputs = model(inputs)
    print(outputs)



