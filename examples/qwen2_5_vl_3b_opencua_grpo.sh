set -x

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m verl.trainer.main \
    config=examples/opencua_config.yaml \
    data.train_files=datasets/opencua_dataset.json@train \
    data.val_files=datasets/opencua_dataset_test.json@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen2_5_vl_3b_opencua_grpo \
    trainer.n_gpus_per_node=8