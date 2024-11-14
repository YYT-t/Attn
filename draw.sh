export CUDA_VISIBLE_DEVICES="1"

for n_lh in 3 #8 6 3  #12 6 3 1  #2 4 6
do
for num_traces in  10000  #2000 4000 6000 8000
do
for max_e in 5
do
python draw.py --mode main \
               --max_examples $max_e\
               --vocab_size 52\
               --env_val_num_low 10\
               --chain_val_num 50\
               --leak_prob_node 0\
               --leak_prob_val 0\
                --tl_low 4\
                --addlen 5\
                --context_lower 1\
                --context_upper 7\
                --context_div 7\
               --n_layers $n_lh\
                --n_heads $n_lh\
                --hidden_size 720\
                --train_epoch 20 \
                --save_steps 20000\
                --probe_mean_num 10\
                --if_in_colab n
done
done
done
done
done
done


##$(($num_traces+$num_traces)) \