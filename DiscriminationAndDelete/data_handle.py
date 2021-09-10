import pandas as pd
import random


def read_douban_dialog():
    """read Sentimental Douban Conversation Corpus"""
    df = pd.read_excel('data/SentimentalDoubanConversationCorpus/douban_sentimental.xlsx')
    d_id = list(df["d_id"])
    label = list(df["label"])
    text = list(df["text"])

    print(len(label))
    print(len(text))
    print(len(set(d_id)))

    # random select 150  for dev
    random.seed(10)
    d_index = random.sample(set(d_id), 150)
    print(len(d_index))
    print(len(set(d_index)))
    train_text = []
    train_label = []
    dev_text = []
    dev_label = []
    dev_d_id = []
    for i in range(len(text)):
        if label[i] != 3:
            if d_id[i] in d_index:
                dev_text.append(str(text[i]))
                dev_label.append(label[i])
                dev_d_id.append(d_id[i])
            else:
                train_text.append(str(text[i]))
                train_label.append(label[i])
    print(len(set(dev_d_id)))
    print(len(train_text))
    print(len(dev_text))
    print(set(train_label))
    print(set(dev_label))

    # analysis
    value = [train_label, dev_label, label]
    for i in range(len(value)):
        print("0:{}, 1:{}, 2:{}, 3:{}".format(value[i].count(0),
                                              value[i].count(1),
                                              value[i].count(2),
                                              value[i].count(3)))
    print(train_text[0:2])
    print(dev_text[0:2])

    return train_text, train_label, dev_text, dev_label, dev_d_id


def get_douban_test_responses():
    """read Douban Conversation Corpus test responses"""
    res = []
    with open("data/DoubanConversationCorpus/test.txt", "r") as f:
        for i, line in enumerate(f.readlines()):
            res.append(line.strip().split("\t")[-1])
    print(len(res))
    print(res[0:20])
    return res


if __name__ == "__main__":
    read_douban_dialog()
    get_douban_test_responses()
