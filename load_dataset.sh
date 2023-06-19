export PYTHONPATH=utils/gpt-neox/
range=13000
for iter in {13000..143000..13000}
do
    python utils/batch_viewer.py \
        --start_iteration $((iter-range)) \
        --end_iteration $iter \
        --mode save \
        --conf_dir utils/dummy_config.yml \
        --save_path "data/checkpoint_${iter}/"
done
