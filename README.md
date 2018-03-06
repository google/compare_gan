## Compare GAN code.

This is the code that was used in "Are GANs Created Equal? A Large-Scale Study"
paper (https://arxiv.org/abs/1711.10337).

### Installation:

To install, run:

```shell
python -m pip install -e . --user
```

After installing, make sure to run

```shell
compare_gan_prepare_datasets.sh
```

It will download all the necessary datasets and frozen TF graphs. By default it will store them in ``/tmp/datasets``.

### Running

compare_gan has two binaries:

  * ``generate_tasks`` - that creates a list of files with parameters to execute
  * ``run_one_task`` - that executes a given task, both training and evaluation, and stores results in the CSV file.


```shell
# Create tasks for experiment "test" in directory /tmp/results. See "src/generate_tasks_lib.py" to see other possible experiments.
compare_gan_generate_tasks --workdir=/tmp/results --experiment=test

# Run task 0 (training and eval)
compare_gan_run_one_task --workdir=/tmp/results --task_num=0 --dataset_root=/tmp/datasets

# Run task 1 (training and eval)
compare_gan_run_one_task --workdir=/tmp/results --task_num=1 --dataset_root=/tmp/datasets
```

Results (FID and inception scores for checkpoints) will be stored in ``/tmp/results/TASK_NUM/scores.csv``.
