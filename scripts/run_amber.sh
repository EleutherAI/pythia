for j in {110..120}
do
    for i in {0..15}
    do
        sbatch single_runner_amber.sbatch $j $i
    done
done
