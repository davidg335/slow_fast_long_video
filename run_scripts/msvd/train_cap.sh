torchrun --nproc_per_node=4 \
    --master_port=34655 \
    train.py \
    --cfg-path lavis/projects/malmm/cap_msvd.yaml \
    --options \
    model.arch blip2_vicuna_instruct \
    model.model_type vicuna7b \
    model.load_finetuned False \
    model.load_pretrained True \
    model.num_query_token 32 \
    model.vit_precision fp16 \
    model.freeze_vit True \
    model.memory_bank_length 40 \
    model.num_frames 80 \
    run.init_lr 1e-5 \
    run.max_epoch 10 \
    run.num_beams 5 \
    run.batch_size_train 16 \
    run.batch_size_eval 16 \
    run.accum_grad_iters 2 \
    run.num_workers 10 \
    run.seed 42 \
    run.evaluate False \
    run.valid_splits "['val']" \
    run.report_metric True \
    run.prefix train
    # run.resume_ckpt_path
