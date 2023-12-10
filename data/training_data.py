import numpy as np

data = np.array([
    [64, 188],
    [72, 186],   
    [70, 183], 
    [67, 176], 
    [63, 174], 
    [75, 190], 
    [70, 186], 
    [60, 186], 
    [50, 156], 
    [65, 186], 
    [57, 166], 
    [64, 170], 
    [54, 160], 
    [56, 163], 
    [58, 165], 
    [60, 156],
    [65, 175],
    [82, 178],
])

data = data - np.array([60, 175])

all_y_trues = np.array([
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
])

data_2_prototype = np.arange(-1, 1, 0.01)
all_y_trues_2S_prototype = np.square(data_2_prototype)

data_2 = data_2_prototype
all_y_trues_2S = all_y_trues_2S_prototype