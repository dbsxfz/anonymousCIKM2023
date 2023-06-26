# from train_augDF import train_augDF
from train_augDF import train_augDF
import sys


dataset = str(sys.argv[2])

model = str(sys.argv[4])

mode = str(sys.argv[6])

random_state = int(sys.argv[8])

train_augDF(dataset, model, mode, random_state)