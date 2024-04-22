import os

root_path = os.path.abspath(os.path.dirname(__file__))
output_way = 'cls'
ef_construction = 3000  # ef_construction defines a construction time/accuracy trade-off
M = 64
all_simcse_hnsw_path = os.path.join(root_path, 'result/simcse_hnsw/all_simcse_hnsw')
center_simcse_hnsw_path = os.path.join(root_path, 'result/center_hnsw/key_center_simcse_hnsw')
best_center_simcse_hnsw_path = os.path.join(root_path, 'result/best/key_center_simcse_hnsw')
ge2e_center_simcse_hnsw_path = os.path.join(root_path, 'result/ge2e_simcse/key_center_simcse_hnsw')

epoch_5_best_simcse_hnsw_path = os.path.join(root_path, 'result/epoch5_best/key_center_simcse_hnsw')