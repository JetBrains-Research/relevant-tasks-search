from src.models.bert_recommender import BertRecommender
import random
import pickle


def retrieve(bert_rec: BertRecommender):
    with open("./data/val_data.tsv", "r", encoding="utf-8") as fin:
        data = fin.readlines()[1:]
    with open("./data/graph/data.pkl", "rb") as fin:
        id2name = pickle.load(fin)["id2name"]
    random.seed(42)
    random.shuffle(data)
    for i in range(10):
        line = data[i].strip().split("\t")
        test_task = bert_rec.create_task(
            step_id=int(line[0]), topic_id=int(line[1]), preprocessed_text=line[2], raw_text=line[3]
        )
        tasks = bert_rec.retrieve(test_task)
        fout = open("./relevant.tsv", "a", encoding="utf-8")
        fout.write(
            "\t".join(
                [
                    test_task.raw_text.strip(),
                    str(test_task.step_id),
                    str(test_task.topic_id),
                    id2name[test_task.topic_id],
                ]
            )
            + "\n"
        )
        for j, rel_task in enumerate(tasks):
            fout.write(
                "\t".join(
                    [
                        rel_task[1].raw_text.strip(),
                        str(rel_task[1].step_id),
                        str(rel_task[1].topic_id),
                        id2name[rel_task[1].topic_id],
                        str(rel_task[2]),
                    ]
                )
                + "\n"
            )
        fout.close()


if __name__ == "__main__":
    bert_rec = BertRecommender(from_file=False)
    bert_rec.train(path_to_train="./data/train_data.tsv")
    print(bert_rec.evaluate(path_to_test="./data/test_data.tsv", t=1))
    retrieve(bert_rec=bert_rec)
