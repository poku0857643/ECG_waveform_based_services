
import pandas as pd
import numpy as np


from datasets import datasets

df = datasets.load("iris")

x = df[['sepal_length', 'petal_length']]
y = df[['petal_length']]

