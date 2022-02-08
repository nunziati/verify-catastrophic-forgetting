import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib as mpl
import matplotlib.pyplot as plt
from dataset import *
from nets import *
from classifier import *
import time
from datetime import datetime
import PySimpleGUI as sg
import argparse

if __name__ == "__main__":
    layout=[[]]
    sg.Window(title="Launcher", margins=(100, 50))