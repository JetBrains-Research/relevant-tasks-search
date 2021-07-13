from src.models.bert_recommender import BertRecommender
import random


def retrieve(bert_rec: BertRecommender):
    with open("./data/val_data.tsv", "r", encoding="utf-8") as fin:
        data = fin.readlines()[1:]
    random.seed(42)
    random.shuffle(data)
    for i in range(10):
        line = data[i].strip().split("\t")
        test_task = bert_rec.create_task(
            step_id=int(line[0]), topic_id=int(line[1]), preprocessed_text=line[2], raw_text=line[3]
        )
        tasks = bert_rec.retrieve(test_task)
        fout = open("./retrieval.txt", "a", encoding="utf-8")
        fout.write(f"Given task:\n{test_task.raw_text}\nTopic id: {test_task.topic_id}\nStep id: {test_task.step_id}\n")
        for j, rel_task in enumerate(tasks):
            fout.write(
                f"Top {j + 1}:\nTopic id: {rel_task[1].topic_id}\nStep id: {rel_task[1].step_id}\nDistance in graph: {rel_task[2]}\n{rel_task[1].raw_text}\n\n"
            )
        fout.close()


if __name__ == "__main__":
    bert_rec = BertRecommender(from_file=False)
    bert_rec.train(path_to_train="./data/train_data.tsv")
    print(bert_rec.evaluate(path_to_test="./data/val_data.tsv", t=0))
    retrieve(bert_rec=bert_rec)
