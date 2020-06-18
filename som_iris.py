import numpy as np
import matplotlib.pyplot as plt
import random

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
  qtdWines = 4
  Dim = 13
  Rows = 7; Cols = 9
  RangeMax = Rows + Cols
  LearnMax = 0.1
  StepsMax = 5000  
  e = 0.001
  classes = []
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
      show_matrix2(Rows, Cols, data_x, data_y, MAP_final)
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
      label_map[i][j] = most_common(mapping[i][j], 4)

  for i in range(10):
    t = random.randint(0, len(data_x))
    (bmu_row, bmu_col) = closest_node(data_x, t, map, Rows, Cols)
    print(t)
    print(bmu_row, bmu_col)
    print(label_map[bmu_row][bmu_col])
  # quantidade de cores no mapa
  plt.imshow(label_map, cmap=plt.cm.get_cmap('terrain_r', 5), interpolation='nearest')
  plt.colorbar()
  plt.show()

def show_matrix2(rows, cols, data_x, data_y, map_final):
    BMU = np.zeros([2],dtype=np.int32)
    result_map = np.zeros([rows,cols,3],dtype=np.float32)

    i=0
    for d in data_x:
    
        pattern_ary = np.tile(d, (rows, cols, 1))
        Eucli_MAP = np.linalg.norm(pattern_ary - map_final, axis=2)

    # Get the best matching unit(BMU) which is the one with the smallest Euclidean distance
        BMU = np.unravel_index(np.argmin(Eucli_MAP, axis=None), Eucli_MAP.shape)
    
        x = BMU[0]
        y = BMU[1]
    
        if data_y[i] == 0:
            if result_map[x][y][0] < 1.0:
                result_map[x][y] += np.asarray([0.2,0,0])
        elif data_y[i] == 1:
             if result_map[x][y][1] < 1.0:
                result_map[x][y] += np.asarray([0,0.2,0])
        elif data_y[i] == 2:
            if result_map[x][y][2] < 1.0:
                 result_map[x][y] += np.asarray([0,0,0.2])
        i+=1
        result_map = np.flip(result_map,0)
    
    #print result_map
    print(result_map)

    plt.imshow(result_map, interpolation='nearest')
    plt.colorbar()
    plt.show()
# ==================================================================

if __name__=="__main__":
  main()
