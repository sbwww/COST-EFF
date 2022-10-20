# Datasets

You can download and process GLUE datasets from

- [Hugging Face](https://huggingface.co/datasets/glue "https://huggingface.co/datasets/glue") (Hugging Face `datasets` library required)
- [W4ngatang/download_glue_data.py](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e "https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e")

## Notice

`run.sh` and `test.sh` contain path to datasets like

```sh
    --data_dir ./data/glue/${TASK_NAME} \
```

Please don't forget to change the path and naming if you want to use other datasets.
