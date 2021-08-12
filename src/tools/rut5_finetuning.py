import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import DatasetDict, load_metric
import nltk


def read_data(input_path: str, load_courses=False):
    """
    Used for reading data from input path and returning pd.DataFrames
    :param input_path: path to dir with train/test/val .csv files
    :param load_courses: if True, loading data about courses and sections too
    :return:
    """
    if not isinstance(input_path, str):
        raise Exception(f"You passed {type(input_path)} instead of str type")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} doesn't exist")

    data_train = pd.read_csv(input_path + "/train.csv")
    data_val = pd.read_csv(input_path + "/val.csv")
    data_test = pd.read_csv(input_path + "/test.csv")

    data_train = data_train.dropna().reset_index(drop=True)
    data_val = data_val.dropna().reset_index(drop=True)
    data_test = data_test.dropna().reset_index(drop=True)

    if load_courses:
        data_courses = pd.read_csv(input_path + "/popular_courses.csv")
        data_sections = pd.read_csv(input_path + "/popular_courses_sections.csv")
        return data_train, data_val, data_test, data_courses, data_sections
    else:
        return data_train, data_val, data_test


def create_dataset(input_path: str, output_path: str):
    """
    Used for dataset's creation. You should provide input path with train/test/val .csv files and
    sections & courses data
    :param input_path: path to load train/test/val .csv files and sections & courses data
    :param output_path: path to save datasets
    :return:
    """
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    data_train, data_val, data_test, data_courses, data_sections = read_data(input_path, load_courses=True)

    dataset_train = pd.DataFrame({"text": data_train.preprocessed_text, "id": data_train.id})
    dataset_test = pd.DataFrame({"text": data_test.preprocessed_text, "id": data_test.id})
    dataset_val = pd.DataFrame({"text": data_val.preprocessed_text, "id": data_val.id})

    course2title = {}
    for index in data_courses.index:
        course2title[data_courses.at[index, "id"]] = data_courses.at[index, "title"]

    section2title = {}
    for index in data_sections.index:
        section2title[data_sections.at[index, "id"]] = data_sections.at[index, "title"]

    # Adding key word "summarize" which T5 is needed
    dataset_train.text = "summarize: " + dataset_train.text + " </s>"
    dataset_test.text = "summarize: " + dataset_test.text + " </s>"
    dataset_val.text = "summarize: " + dataset_val.text + " </s>"

    # Adding some king of summary - section_name + delimiter + course_name
    dataset_train["summary"] = ""
    dataset_test["summary"] = ""
    dataset_val["summary"] = ""
    for index in data_train.index:
        course_id = data_train.at[index, "course_id"]
        section_id = data_train.at[index, "section_id"]
        dataset_train.at[index, "summary"] = section2title[section_id] + " в курсе " + course2title[course_id]
    for index in data_test.index:
        course_id = data_test.at[index, "course_id"]
        section_id = data_test.at[index, "section_id"]
        dataset_test.at[index, "summary"] = section2title[section_id] + " в курсе " + course2title[course_id]
    for index in data_val.index:
        course_id = data_val.at[index, "course_id"]
        section_id = data_val.at[index, "section_id"]
        dataset_val.at[index, "summary"] = section2title[section_id] + " в курсе " + course2title[course_id]

    # Saving datasets to the output dir
    dataset_train.to_csv(output_path + "/dataset_train.csv", index=False, header=True)
    dataset_test.to_csv(output_path + "/dataset_test.csv", index=False, header=True)
    dataset_val.to_csv(output_path + "/dataset_val.csv", index=False, header=True)


def preprocess_function(examples, tokenizer, max_length=512):
    """
    This function is used to tokenized text in datasets
    Adopt by me almost without changes from https://github.com/huggingface/notebooks/blob/master/examples/summarization.ipynb
    :param examples: dataset
    :param tokenizer: tokenizer to use
    :param max_length: max length of sequence in tokens
    :return:
    """
    inputs = [doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding="max_length")

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=max_length, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred, tokenizer, metric):
    """
    This function is used to compute ROUGE metric.
    Adopt by me almost without changes from https://github.com/huggingface/notebooks/blob/master/examples/summarization.ipynb
    :param eval_pred:
    :param tokenizer:
    :param metric:
    :return:
    """
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


def train(
    input_path: str, output_path: str, metric_name: str = "rouge", model_checkpoint: str = "cointegrated/rut5-small"
):
    """
    Used for training. You should provide input path with already computed datasets
    :param input_path: path to directory with datasets
    :param output_path: where to save trained model
    :param metric_name: metric to compute while training
    :param model_checkpoint: checkpoint to start finetuning
    :return:
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} doesn't exist")
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    raw_datasets = DatasetDict.from_csv(
        {
            "train": input_path + "/dataset_train.csv",
            "test": input_path + "/dataset_test.csv",
            "validation": input_path + "/dataset_val.csv",
        }
    )
    metric = load_metric(metric_name)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    tokenized_datasets = raw_datasets.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    batch_size = 4
    args = Seq2SeqTrainingArguments(
        output_path + "/rut5_finetuned",
        evaluation_strategy="steps",
        eval_steps=200,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=1e-5,
        save_total_limit=1,
        save_strategy="steps",
        save_steps=1000,
        num_train_epochs=2,
        predict_with_generate=True,
        logging_steps=200,
        logging_first_step=True,
        logging_dir=output_path + "/logs",
        push_to_hub=False,
    )

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda x: compute_metrics(x, tokenizer, metric),
    )

    trainer.train()

    model.save_pretrained(output_path + "/rut5_finetuned")


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-t", "--task", choices=["create_dataset", "train"], required=True)
    arg_parser.add_argument("-o", "--output", required=True, help="Path to output directory")
    arg_parser.add_argument("-i", "--input", required=True, help="Path to input directory")
    return arg_parser


if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    if __args.task == "create_dataset":
        create_dataset(__args.input, __args.output)
    elif __args.task == "train":
        train(__args.input, __args.output)
