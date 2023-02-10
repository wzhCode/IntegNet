
import os
import argparse
import time
import tensorflow as tf

from utils.math_graph import *
from data_loader.data_utils import *
from models.trainer import model_train
from models.tester import model_test

# gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# gpu configuration
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)

# cpu
#os.environ["KMP_WARNINGS"] = "0"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
#os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# parameter configuration
parser = argparse.ArgumentParser()
parser.add_argument('--n_dcpair', type=int, default=330)
parser.add_argument('--n_his', type=int, default=12)
parser.add_argument('--n_pred', type=int, default=1)
parser.add_argument('--ks', type=int, default=3)
parser.add_argument('--kt', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--save', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-3)

args = parser.parse_args()
print(f'Parameter configs: {args}')


# Load graph wighted adjacency matrix W
W = weight_matrix('file_path_weight_matrix')  # your file path of weighted adjacency matrix W
print('W shape: ', W.shape)

# Calculate graph kernel
L = scaled_laplacian(W)
print('L shape: ', L.shape)

# Chebyshev polynomials approximation
Lk = cheb_poly_approx(L, args.ks, args.n_dcpair)
tf.add_to_collection(name='graph_kernel', value=tf.cast(tf.constant(Lk), tf.float32))
print('LK shape: ', Lk.shape)

# TCN+GCN blocks configuration: the input/output channel size in each layer
#blocks = [[1, 16, 16]] # 1 block
#blocks = [[1, 32, 32]] # 1 block
blocks = [[1, 64, 64]] # 1 block
#blocks = [[1, 16, 16],[16, 32, 32]] # 2 blocks
#blocks = [[1, 32, 32],[32, 64, 64]] # 2 blocks
#blocks = [[1, 16, 16],[16, 32, 32],[32, 64, 64]] # 3 blocks

# Input
n_train, n_val, n_test = 14, 2, 2  # days
day_slot = 288  # time intervals per day
data_file_path = 'file_path_traffic_data'  # your file path of traffic data

dataDC = data_gen(data_file_path, (n_train, n_val, n_test), args.n_dcpair, args.n_his, args.n_pred, day_slot)


# Output
name = 'integnet_' + str(len(blocks)) +'l_' + 'c' + str(blocks[0][1]) + '_' + args.train_mode
results_file_path = './output/pred_res_' + name + '.npy'  # save predict results
save_model_path = './output/models/'
save_model_name = save_model_path + name  # save trained model


if __name__ == '__main__':
    print('Training model:')
    start = time.time()
    model_train(dataDC, blocks, args, save_model_name)
    end = time.time()
    run_time = end - start
    print('Training time: ', run_time)
    print('Inference:')
    start = time.time()
    model_test(dataDC, dataDC.get_len('test'), args.n_his, args.n_pred, results_file_path, save_model_path)
    end = time.time()
    run_time = end - start
    print('Inference time: ', run_time)

