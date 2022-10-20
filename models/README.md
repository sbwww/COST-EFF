# Models

You can download pre-trained models from [Hugging Face](https://huggingface.co/bert-base-uncased "https://huggingface.co/bert-base-uncased")

## Example

```sh
ls ./pretrained_model/bert-base-uncased
config.json        tokenizer_config.json  vocab.txt
pytorch_model.bin  tokenizer.json
```

## Some useful models

- [BERT (Devlin et al., 2019)](https://huggingface.co/bert-base-uncased "https://huggingface.co/bert-base-uncased"): Original BERT model.
- [BERT Miniatures (Turc et al., 2019)](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2 "https://huggingface.co/google/bert_uncased_L-2_H-128_A-2"): A collection of 24 BERT with different sizes.
- [TinyBERT (Jiao et al., 2020)](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D "https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D"): A strong baseline of BERT compression with General Distillation + Task-specific Distillation + Data Augmentation.
- [ElasticBERT (Liu et al., 2022)](https://huggingface.co/fnlp/elasticbert-base "https://huggingface.co/fnlp/elasticbert-base"): A strong baseline and backbone model of multi-exit language model.

## Notice

`run.sh` and `test.sh` contain path to models like

```sh
    --teacher_model ./models/finetuned_model/${TASK_NAME} \
    --student_model ./models/pretrained_model/bert-base-uncased \
```

and in `Costeff/run_glue_costeff.py`

```py
    tokenizer = BertTokenizer.from_pretrained("./models/pretrained_model/bert-base-uncased",
                                              do_lower_case=args.do_lower_case)
```

Please don't forget to change the path and naming if you want to use other models.
