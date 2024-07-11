for i in {100..120}
do
    echo $i
    ID=$i python preload_data_models.py
done