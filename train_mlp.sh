export CUDA_VISIBLE_DEVICES="8"
for num_traces in 10000 
do
for nl in  2 
do
for hidden_size in 1000
do
for max_e in 5
do
for len in 5 
do
for width in 2
do 
for child_len in 3
do
for wind in 200
do
for gt in 0
do
python main_mlp.py --num_icl_train_traces 5000 \
               --num_mk_train_trace $(($num_traces+$num_traces)) \
               --max_examples $max_e\
               --graph_type $gt\
               --graph_len $len\
               --graph_width $width\
               --max_child_chain_len $child_len\
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
               --n_layers $nl\
                --hidden_size $hidden_size\
                --window_size $wind\
                --if_train y\
                --train_epoch 20 \
                --save_steps 2780\
                --if_test y\
                --if_plot n \
                --if_probe n \
                --test_epoch 9\

done
done
done
done
done
done
done
done
done