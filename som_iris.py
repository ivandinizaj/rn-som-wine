
## som_iris.py
## SOM for Iris dataset
## Anaconda3 5.2.0 (Python 3.6.5)

## ==================================================================
#import matplotlib.pyplot as plt
#import numpy as np
#import math
## note: if this fails, try >pip uninstall matplotlib
## and then >pip install matplotlib
#def closest_node(data, t, map, m_rows, m_cols):
#  # (row,col) of map node closest to data[t]
#  result = (0,0)
#  small_dist = 1.0e20
#  for i in range(m_rows):
#    for j in range(m_cols):
#      ed = euc_dist(map[i][j], data[t])
#      if ed < small_dist:
#        small_dist = ed
#        result = (i, j)
#  return result

## 5.1,3.5,1.4,0.2,1
## 4.7,3.2,1.3,0.2,1

## [0.4; 0.3; 0.1; 0;0]

#def euc_dist(v1, v2):
#  return np.linalg.norm(v1 - v2)

#def manhattan_dist(r1, c1, r2, c2):
#  return np.abs(r1 - r2) + np.abs(c1 - c2)

#def most_common(lst, n):
#  # lst is a list of values 0 .  .  n
#  if len(lst) == 0: return -1
#  counts = np.zeros(shape=n, dtype=np.int)
#  for i in range(len(lst)):
#    counts[lst[i]] += 1
#  return np.argmax(counts)

## ==================================================================
#def main():
#  # 0.  get started
#  np.random.seed(1)
#  Dim = 13
#  Rows = 10
#  Cols = 10
#  RangeMax = Rows + Cols
#  LearnMax = 0.5
#  StepsMax = 5000

#  # 1.  load data
#  print("\nLoading Wine data into memory \n")
#  data_file = "wine.data.txt"
#  data_x = np.loadtxt(data_file, delimiter=",", usecols=range(1,Dim + 1),
#    dtype=np.float64)
#  data_y = np.loadtxt(data_file, delimiter=",", usecols=[0],
#    dtype=np.int)
#  print("\nLoading Wine data completed \n")

#  # normalize data
#  for i in range(len(data_y)):
#      data_y[i] = data_y[i] - 1

#  # 2.  construct the SOM
#  print(f'Constructing a {Rows}x{Cols} SOM from the wine data')
#  map = np.random.random_sample(size=(Rows,Cols,Dim))
#  for s in range(StepsMax):
#    if s % (StepsMax / 20) == 0: 
#      #show_matrix(Rows, Cols, data_x, data_y, map, Dim)
#      print("step = ", str(s))

#    pct_left = 1.0 - ((s * 1.0) / StepsMax)
#    curr_rate = LearnMax*(1-(float(s)/StepsMax))
#    curr_range = RangeMax*math.exp(- float(s)/StepsMax)
#    #curr_range = RangeMax * math.exp(- float(s)/StepsMax)
#    #curr_rate = LearnMax * math.exp(- float(s)/StepsMax)

#    # print(curr_rate)

#    t = np.random.randint(len(data_x))
#    # print(t)

#    # encontrei o vinho mais proximo do T
#    (bmu_row, bmu_col) = closest_node(data_x, t, map, Rows, Cols)
#    for i in range(Rows):
#      for j in range(Cols):
#        if manhattan_dist(bmu_row, bmu_col, i, j) < curr_range:
#          map[i][j] = map[i][j] + curr_rate * (data_x[t] - map[i][j])

#    #print(map)
#  show_matrix(Rows, Cols, data_x, data_y, map, Dim)

#  print("SOM construction complete \n")

#  ## # 3. construct U-Matrix
#  #print("Constructing U-Matrix from SOM")
#  #u_matrix = np.zeros(shape=(Rows,Cols), dtype=np.float64)
#  #for i in range(Rows):
#  #  for j in range(Cols):
#  #    v = map[i][j]  # a vector 
#  #    sum_dists = 0.0; ct = 0
     
#  #    if i-1 >= 0:    # above
#  #      sum_dists += euc_dist(v, map[i-1][j]); ct += 1
#  #    if i+1 <= Rows-1:   # below
#  #      sum_dists += euc_dist(v, map[i+1][j]); ct += 1
#  #    if j-1 >= 0:   # left
#  #      sum_dists += euc_dist(v, map[i][j-1]); ct += 1
#  #    if j+1 <= Cols-1:   # right
#  #      sum_dists += euc_dist(v, map[i][j+1]); ct += 1
      
#  #    u_matrix[i][j] = sum_dists / ct
#  #print("U-Matrix constructed \n")

#  # # display U-Matrix
#  #plt.imshow(u_matrix, cmap='gray')  # black = close = clusters
#  #plt.show()

#  # 4. because the data has labels, another possible visualization:
#  # associate each data label with a map node
#def show_matrix(Rows, Cols, data_x, data_y, map, Dim):
#  print("Associando cada rotulo de dados a um no do mapa")
#  mapping = np.empty(shape=(Rows,Cols), dtype=object)
#  for i in range(Rows):
#    for j in range(Cols):
#      mapping[i][j] = []

#  for t in range(len(data_x)):
#    (m_row, m_col) = closest_node(data_x, t, map, Rows, Cols)
#    mapping[m_row][m_col].append(data_y[t])

#  label_map = np.zeros(shape=(Rows,Cols), dtype=np.int)
#  for i in range(Rows):
#    for j in range(Cols):
#      label_map[i][j] = most_common(mapping[i][j], Dim - 1)

#  # quantidade de cores no mapa
#  plt.imshow(label_map, cmap=plt.cm.get_cmap('terrain_r', 4))
#  plt.colorbar()
#  plt.show()
  
## ==================================================================

#if __name__ == "__main__":
#  main()

# som_iris.py
# SOM for Iris dataset
# Anaconda3 5.2.0 (Python 3.6.5)

# ==================================================================

import numpy as np
import matplotlib.pyplot as plt
# note: if this fails, try >pip uninstall matplotlib
# and then >pip install matplotlib

def closest_node(data, t, map, m_rows, m_cols):
  # (row,col) of map node closest to data[t]
  result = (0,0)
  small_dist = 1.0e20
  for i in range(m_rows):
    for j in range(m_cols):
      ed = euc_dist(map[i][j], data[t])
      if ed < small_dist:
        small_dist = ed
        result = (i, j)
  return result

def euc_dist(v1, v2):
  return np.linalg.norm(v1 - v2) 

def manhattan_dist(r1, c1, r2, c2):
  return np.abs(r1-r2) + np.abs(c1-c2)

def most_common(lst, n):
  # lst is a list of values 0 . . n
  if len(lst) == 0: return -1
  counts = np.zeros(shape=n, dtype=np.int)
  for i in range(len(lst)):
    counts[lst[i]] += 1
  return np.argmax(counts)

# ==================================================================

def main():
  # 0. get started
  np.random.seed(1)
  qtdWines = 3
  Dim = 13
  Rows = 10; Cols = 10
  RangeMax = Rows + Cols
  LearnMax = 0.5
  StepsMax = 5000  
  e = 0.001
  # 1. load data
  print("\nLoading Iris data into memory \n")
  data_file = "wine.data.txt"
  data_x = np.loadtxt(data_file, delimiter=",", usecols=range(1,Dim + 1),
    dtype=np.float64)
  data_y = np.loadtxt(data_file, delimiter=",", usecols=[0],
    dtype=np.int)
  # normalize data
  for i in range(len(data_y)):
      data_y[i] = data_y[i] - 1
  
  # 2. construct the SOM
  print("Constructing a 30x30 SOM from the iris data")

  #map1 = np.random.random_sample(size=(Rows,Cols,Dim))
  map = np.random.uniform(size=(Rows,Cols,Dim))
  prev_MAP = np.zeros((Rows,Cols,Dim))
  convergence = [1]
  timestep=1
  flag=0
  for s in range(StepsMax):
    J = np.linalg.norm(map - prev_MAP)
    prev_MAP = np.copy(map)
    if J < e:
      show_matrix(Rows, Cols, data_x, data_y, MAP_final, Dim)
      flag = 1
      break
    if s % (StepsMax/10) == 0: print("step = ", str(s))
    pct_left = 1.0 - ((s * 1.0) / StepsMax)
    curr_range = (int)(pct_left * RangeMax)
    curr_rate = pct_left * LearnMax

    t = np.random.randint(len(data_x))
    (bmu_row, bmu_col) = closest_node(data_x, t, map, Rows, Cols)
    for i in range(Rows):
      for j in range(Cols):
        if manhattan_dist(bmu_row, bmu_col, i, j) < curr_range:
          map[i][j] = map[i][j] + curr_rate * (data_x[t] - map[i][j])
    if J < min(convergence):
        print('Lower error found: %s' %str(J) + ' at epoch: %s' % str(s))
        print('\tLearning rate: ' + str(curr_rate))
        print('\tNeighbourhood radius: ' + str(curr_range))
        MAP_final = map
    convergence.append(J)
  print("SOM construction complete \n")

  # 3. construct U-Matrix
  # print("Constructing U-Matrix from SOM")
  # u_matrix = np.zeros(shape=(Rows,Cols), dtype=np.float64)
  # for i in range(Rows):
  #   for j in range(Cols):
  #     v = map[i][j]  # a vector 
  #     sum_dists = 0.0; ct = 0
     
  #     if i-1 >= 0:    # above
  #       sum_dists += euc_dist(v, map[i-1][j]); ct += 1
  #     if i+1 <= Rows-1:   # below
  #       sum_dists += euc_dist(v, map[i+1][j]); ct += 1
  #     if j-1 >= 0:   # left
  #       sum_dists += euc_dist(v, map[i][j-1]); ct += 1
  #     if j+1 <= Cols-1:   # right
  #       sum_dists += euc_dist(v, map[i][j+1]); ct += 1
      
  #     u_matrix[i][j] = sum_dists / ct
  # print("U-Matrix constructed \n")

  # # display U-Matrix
  # plt.imshow(u_matrix, cmap='gray')  # black = close = clusters
  # plt.show()

def show_matrix(Rows, Cols, data_x, data_y, map, Dim):
  print("Associando cada rotulo de dados a um no do mapa")
  mapping = np.empty(shape=(Rows,Cols), dtype=object)
  for i in range(Rows):
    for j in range(Cols):
      mapping[i][j] = []

  for t in range(len(data_x)):
    (m_row, m_col) = closest_node(data_x, t, map, Rows, Cols)
    mapping[m_row][m_col].append(data_y[t])

  label_map = np.zeros(shape=(Rows,Cols), dtype=np.int)
  for i in range(Rows):
    for j in range(Cols):
      label_map[i][j] = most_common(mapping[i][j], Dim - 1)

  # quantidade de cores no mapa
  plt.imshow(label_map, cmap=plt.cm.get_cmap('terrain_r', 4))
  plt.colorbar()
  plt.show()

# ==================================================================

if __name__=="__main__":
  main()
