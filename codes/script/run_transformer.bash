cd ..

mkdir -p ./tmp/

# for multiple GPU, use 0,1 not fully supported yet
CUDA_VISIBLE_DEVICES=0 python recsys_main.py \
    --output_dir "./tmp/" \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --data_path "~/dataset/sessions_with_neg_samples_example/" \
    --per_device_train_batch_size 128 \
    --model_type "xlnet" \
    --loss_type "margin_hinge" \
    --fast_test
cd -

# model_type: xlnet, gpt2, longformer
# loss_type: cross_entropy, hinge_loss
# fast-test: quickly finish loop over examples to check code after the loop