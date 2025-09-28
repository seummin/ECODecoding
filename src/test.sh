for lr in 1e-5
do
    for acc in 8
    do
        for lam in 1.6
        do
            for sf in 0.1
            do
            CUDA_VISIBLE_DEVICES=1 python main.py epochs=30 \
            learning_rate=${lr} gradient_accumulation_steps=${acc} method=gpt2_director2 \
            datamodule.data_name=emo seed=616 wandb=False \
            learner.smoothing_factor=${sf} learner.condition_lambda=${lam} learner.use_prompt=True learner.num_attribute=1 learner.pre_seq_len=200 test=True
            done
        done
    done
done