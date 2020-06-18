import numpy as np
import matplotlib.pyplot as plt
import random

def closest_node(data, t, map, m_map_rows, m_cols):
  result = (0,0)
  small_dist = 1.0e20
  for i in range(m_map_rows):
    for j in range(m_cols):
      ed = np.linalg.norm(map[i][j] - data[t]) 
      if ed < small_dist:
        small_dist = ed
        result = (i, j)
  return result

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
  dimensions = 13
  map_rows = 13; 
  map_cols = 13
  range_max = map_rows + map_cols
  learning_max = 0.1
  steps_max = 10000  
  e = 0.001

  # 1. load data
  print("\n\n.::WINES::.\n\n")
  data_file = "wine.data.txt"
  data_values = np.loadtxt(data_file, delimiter=",", usecols=range(1, dimensions + 1),
    dtype=np.float64)
  data_target = np.loadtxt(data_file, delimiter=",", usecols=[0],
    dtype=np.int)

  # normalize data
  for i in range(len(data_target)):
      data_target[i] = data_target[i] - 1
  
  map = np.random.uniform(size=(map_rows,map_cols, dimensions))
  prev_map = np.zeros((map_rows,map_cols, dimensions))
  convergence = [1]

  for s in range(steps_max):
    map_distance = np.linalg.norm(map - prev_map)
    prev_map = np.copy(map)
    
    if map_distance < e:
      show_matrix(map_rows, map_cols, data_values, data_target, SOM, dimensions)
      break

    if s % (steps_max/10) == 0: print("step = " + str(s))
    percent_steps_left = 1.0 - ((s * 1.0) / steps_max)
    current_range = (int)(percent_steps_left * range_max)
    current_rate = percent_steps_left * learning_max

    random_value = np.random.randint(len(data_values))
    (bmu_row, bmu_col) = closest_node(data_values, random_value, map, map_rows, map_cols)
    for i in range(map_rows):
      for j in range(map_cols):
        if manhattan_dist(bmu_row, bmu_col, i, j) < current_range:
          map[i][j] = map[i][j] + current_rate * (data_values[random_value] - map[i][j])

    if map_distance < min(convergence):
        print('Lower error found: %s' %str(map_distance) + ' at step: %s' % str(s))
        print('\tLearning rate: ' + str(current_rate))
        print('\tNeighbourhood radius: ' + str(current_range))
        SOM = map
    convergence.append(map_distance)
 
  print("SOM construction complete \n")

def show_matrix(map_rows, map_cols, data_values, data_target, map, dimensions):
  print("Associando cada rotulo de dados a um no do mapa")
  mapping = np.empty(shape=(map_rows, map_cols), dtype = object)
  for i in range(map_rows):
    for j in range(map_cols):
      mapping[i][j] = []

  for t in range(len(data_values)):
    (m_row, m_col) = closest_node(data_values, t, map, map_rows, map_cols)
    mapping[m_row][m_col].append(data_target[t])

  label_map = np.zeros(shape=(map_rows,map_cols), dtype=np.int)
  for i in range(map_rows):
    for j in range(map_cols):
      label_map[i][j] = most_common(mapping[i][j], 4)

  #Teste
  success = 0
  for i in range(len(data_values)):
    # t = random.randint(0, len(data_values))
    (bmu_row, bmu_col) = closest_node(data_values, i, map, map_rows, map_cols)
    if data_target[i] == label_map[bmu_row][bmu_col]:
      success += 1
    # print(t)
    # print(bmu_row, bmu_col)
    # print(label_map[bmu_row][bmu_col])

  print("\n\nsucesso: ", (float(success)/len(data_values ) )*100 )

  # quantidade de cores no mapa
  plt.imshow(label_map, cmap=plt.cm.get_cmap('terrain_r', 4), interpolation='nearest')
  plt.colorbar()
  plt.show()

# ==================================================================

if __name__=="__main__":
  main()
