for i in {0..15}
do
    sbatch single_runner.sbatch 4000 $i
done
