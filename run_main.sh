python main.py --dataset Cora --num_clients 10 --partition Louvain+ --hid_dim 64 --hops 1 --ipc 1 --personalized --condensing_loop 3000 --local_strategy nd_lp_reg_ft --patience 10 --lr_ft 5e-5 --reg_a 0.2 --reg_max 0.15 --lp_iter 2 --lp_alpha 0.9 --topk 1 --thres 1.0 --d_thres 20 --gpu_id 3

python main.py --dataset CiteSeer --num_clients 10 --partition Louvain+ --hid_dim 64 --hops 1 --ipc 1 --personalized --local_strategy nd_lp_reg_ft --patience 10 --condensing_loop 3000 --lr_ft 5e-5 --reg_a 0.2 --reg_max 0.2 --topk 1 --d_thres 3 --gpu_id 2

python main.py --dataset PubMed --num_clients 10 --partition Louvain+ --hid_dim 64 --hops 1 --ipc 1 --lr_ft 1e-3 --personalized --local_strategy nd_lp_reg_ft --condensing_loop 3000 --reg_a 1.0 --reg_max 0.1 --reg_min 0.05 --lp_iter 4 --lp_alpha 0.6 --topk 1 --gpu_id 0

python main.py --dataset ogbn-arxiv --num_clients 10 --partition Louvain+ --hops 1 --lr_feat 0.05 --lr_adj 0.01 --rate 0.0025 --hid_dim 64 --lr_validation_model 5e-4 --lr_ft 5e-3 --personalized --condensing_loop 2000 --local_strategy nd_lp_reg_ft --reg_a 0.5 --reg_max 0.15 --topk 4 --d_thres 15 --gpu_id 1



python main.py --dataset Cora --num_clients 10 --partition Metis+ --hid_dim 64 --hops 1 --ipc 1 --personalized --condensing_loop 3000 --local_strategy nd_lp_reg_ft --patience 10 --lr_ft 5e-5 --reg_a 0.2 --reg_max 0.2 --lp_iter 2 --lp_alpha 0.9 --topk 1 --gpu_id 1

python main.py --dataset CiteSeer --num_clients 10 --partition Metis+ --hid_dim 64 --hops 1 --ipc 1 --personalized --local_strategy nd_lp_reg_ft --patience 10 --condensing_loop 3000 --lr_ft 5e-5 --reg_a 0.2 --reg_max 0.15 --reg_min 0.05 --topk 1 --d_thres 3 --gpu_id 4

python main.py --dataset PubMed --num_clients 10 --partition Metis+ --hid_dim 64 --hops 1 --ipc 1 --lr_ft 2e-3 --personalized --local_strategy nd_lp_reg_ft --condensing_loop 3000 --reg_a 5.0 --reg_max 0.2 --reg_min 0.05 --lp_iter 4 --lp_alpha 0.6 --topk 1 --gpu_id 1

python main.py --dataset ogbn-arxiv --num_clients 10 --partition Metis+ --hops 1 --lr_feat 0.05 --lr_adj 0.01 --rate 0.0025 --hid_dim 64 --lr_validation_model 5e-4 --lr_ft 5e-3 --personalized --condensing_loop 2000 --local_strategy nd_lp_reg_ft --reg_a 1 --reg_max 0.15 --lp_iter 5 --lp_alpha 0.6 --topk 4 --d_thres 5 --gpu_id 0