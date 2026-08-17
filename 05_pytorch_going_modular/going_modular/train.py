"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch

from timeit import default_timer as timer
from torchvision import transforms
# from going_modular import data_setup #as data_setup in same directory(going_modular) import data_setip also works
import data_setup, engine, model_builder, utils

# Setup hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 32
HIDDEN_UNITS = 16
LEARNING_RATE = 0.001

# Setup directories
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# Create DataLoader's and get class_names
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir = train_dir,
                                                                               test_dir = test_dir,
                                                                               transform = data_transform,
                                                                               batch_size = BATCH_SIZE)

# Create model
model = model_builder.TinyVGG(input_shape = 3,
                              hidden_units = HIDDEN_UNITS,
                              output_shape = len(class_names)).to(device)

# Setup loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters(),
                             lr = LEARNING_RATE)

# start the timer
start_time = timer()

# Start training with help from engine.py
engine.train(model = model,
             train_dataloader = train_dataloader,
             test_dataloader = test_dataloader,
             loss_fn = loss_fn,
             optimizer = optimizer,
             epochs = NUM_EPOCHS,
             device = device)

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time - start_time:.3f} seconds")

# Save the model
utils.save_model(model = model,
                 target_dir = "models",
                 model_name = "05_going_modular_script_mode_tinyvgg_model.pth")
