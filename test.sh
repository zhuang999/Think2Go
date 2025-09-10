CUDA_VISIBLE_DEVICES=0 nohup python eval_vllm.py --model_path ./Llama-3.1-8B  --dataset_name nyc --output_dir ./restart/checkpoint --test_file "datasets/NYC/data/test_codebook_origin.json" > restart/test.txt 2>&1 &  

