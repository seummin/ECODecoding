for lr in 1e-5
do
    for acc in 8
    do
        for ep in 200
        do
        CUDA_VISIBLE_DEVICES=3 python main.py epochs=60 \
        learning_rate=${lr} gradient_accumulation_steps=${acc} method=gpt2_director2 datamodule.data_name=emo seed=616 wandb=False learner.use_prompt=True learner.num_attribute=1 learner.pre_seq_len=${ep}
        done
    done
done