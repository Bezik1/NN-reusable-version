import numpy as np

# data_2_prototype = np.arange(-0.1, 0.1, 0.001)
# all_y_trues_2S_prototype = np.square(data_2_prototype)

# data_2 = data_2_prototype
# all_y_trues_2S = all_y_trues_2S_prototype

x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
y = x**2

data_2_prototype = (x - np.min(x)) / (np.max(x) - np.min(x))
all_y_trues_2S_prototype = (y - np.min(y)) / (np.max(y) - np.min(y)) 