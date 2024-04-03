#!/bin/bash
# python3 -u test_nerf.py --config configs/photoshapes/config.txt --render_test --render_train --load_it 40000  1>photoshapes_testing.log 2>&1
# python3 -u test_nerf.py --config configs/photoshapes/config.txt --render_test --render_train --load_it 450000 --testskip 50000  1>photoshapes_testing.log 2>&1
python3 -u test_nerf.py --config configs/plane_dataset/config.txt --render_test --render_train --load_it 1440000 --testskip 50000 1>plane_testing.log

# make process pool
# processes=()
# for ((i = 20000; i <= 90000; i+=10000)); do
#     command="python3 -u test_nerf.py --config configs/photoshapes/config.txt --render_test --testskip 10000 --load_it ${i}"
#     processes+=("$command")
# done

# # for ((i = 10000; i <= 90000; i+=10000)); do
# #     command="python3 -u test_nerf.py --config configs/photoshapes/config.txt --render_train --trainskip 2000 --load_it ${i}"
# #     processes+=("$command")
# # done

# N=1 # pointless though, 100% gpu util for a single process
# # Function to run processes
# run_processes() {
#     for process in "${@:1:N}"; do
#         # Run the process in the background -> nope
#         $process
#     done

#     # Wait for all background processes to finish
#     wait
# }

# # Determine the number of iterations needed
# num_iterations=$(((${#processes[@]} + 2) / N))

# # Run processes in batches of N
# for ((i = 0; i < num_iterations; i++)); do
#     # Calculate the start index for each batch
#     start_index=$((i * N))

#     # Extract three processes to run in the current batch
#     batch_processes=("${processes[@]:start_index:N}")

#     # Run the processes in the current batch
#     run_processes "${batch_processes[@]}"
# done

