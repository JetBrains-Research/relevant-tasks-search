from src.models.bert_recommender import BertRecommender

if __name__ == '__main__':
    bert_rec = BertRecommender(from_file=False)
    bert_rec.train(path_to_train="./data/train_data.tsv")
    # print(bert_rec.evaluate(path_to_test="./data/val_data.tsv", t=0))
    tasks = bert_rec.retrieve(
        "statements choose correct statements comments java.",
        mode="retrieve")
    for task in tasks:
        print(task[1].topic_id, task[1].raw_text)
