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
from src.tools.sbert_finetuning import read_data


def create_dataset(
    data_train: pd.DataFrame,
    data_val: pd.DataFrame,
    data_test: pd.DataFrame,
    data_courses: pd.DataFrame,
    data_sections: pd.DataFrame,
):

    dataset = pd.DataFrame({"text": data_train.preprocessed_text, "id": data_train.id})
    dataset_test = pd.DataFrame({"text": data_test.preprocessed_text, "id": data_test.id})
    dataset_val = pd.DataFrame({"text": data_val.preprocessed_text, "id": data_val.id})

    course2title = {}
    for index in data_courses.index:
        course2title[data_courses.at[index, "id"]] = data_courses.at[index, "title"]

    section2title = {}
    for index in data_sections.index:
        section2title[data_sections.at[index, "id"]] = data_sections.at[index, "title"]

    dataset.text = "summarize: " + dataset.text + " </s>"
    dataset_test.text = "summarize: " + dataset_test.text + " </s>"
    dataset_val.text = "summarize: " + dataset_val.text + " </s>"

    dataset["summary"] = ""
    dataset_test["summary"] = ""
    dataset_val["summary"] = ""
    for index in data_train.index:
        course_id = data_train.at[index, "course_id"]
        section_id = data_train.at[index, "section_id"]
        dataset.at[index, "summary"] = section2title[section_id] + " в курсе " + course2title[course_id]
    for index in data_test.index:
        course_id = data_test.at[index, "course_id"]
        section_id = data_test.at[index, "section_id"]
        dataset_test.at[index, "summary"] = section2title[section_id] + " в курсе " + course2title[course_id]
    for index in data_val.index:
        course_id = data_val.at[index, "course_id"]
        section_id = data_val.at[index, "section_id"]
        dataset_val.at[index, "summary"] = section2title[section_id] + " в курсе " + course2title[course_id]

    dataset.to_csv("./tmp/dataset_train.csv", index=False, header=True)
    dataset_test.to_csv("./tmp/dataset_test.csv", index=False, header=True)
    dataset_val.to_csv("./tmp/dataset_val.csv", index=False, header=True)


def preprocess_function(examples, tokenizer, max_length=512):
    inputs = [doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding="max_length")

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=max_length, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred, tokenizer, metric):
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


def train():
    raw_datasets = DatasetDict.from_csv(
        {"train": "./dataset_train.csv", "test": "./dataset_test.csv", "validation": "./dataset_val.csv"}
    )
    metric = load_metric("rouge")

    model_checkpoint = "cointegrated/rut5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    tokenized_datasets = raw_datasets.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    batch_size = 4
    args = Seq2SeqTrainingArguments(
        "../data/output_data/rut5_finetuned",
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
        logging_dir="./logs",
        push_to_hub=False,
    )

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

    model.save_pretrained("../data/output_data/rut5_finetuned")


def main():
    paths = ["", "", ""]
    data_train, data_val, data_test = read_data(paths=paths)
    data_courses = pd.read_csv("popular_courses.csv")
    data_sections = pd.read_csv("popular_courses_sections.csv")
    create_dataset(data_train, data_val, data_test, data_courses, data_sections)
    train()


if __name__ == "__main__":
    pass
