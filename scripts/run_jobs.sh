for i in {0..15}
do
    sbatch single_runner.sbatch 42000 $i
done
