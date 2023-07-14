# SST2
SST2_DATASET_PARAMS = {
    "set_name": "sst2",
    "config": None,
    "train_or_test": "train",
    "on_hugging_face": True,
    "content_label_keys": (["sentence"], "label"),
}

# AGNEWS
AGNEWS_DATASET_PARAMS = {
    "set_name": "ag_news",
    "config": None,
    "train_or_test": "train",
    "on_hugging_face": True,
    "content_label_keys": (["text"], "label"),
}

# TREC
TREC_DATASET_PARAMS = {
    "set_name": "trec",
    "config": None,
    "train_or_test": "train",
    "on_hugging_face": True,
    "content_label_keys": (["text"], "coarse_label"),
}

# DBPEDIA
DBPEDIA_DATASET_PARAMS = {
    "set_name": "dbpedia_14",
    "config": None,
    "train_or_test": "train",
    "on_hugging_face": True,
    "content_label_keys": (["content"], "label"),
}

# RTE
RTE_DATASET_PARAMS = {
    "set_name": "glue",
    "config": "rte",
    "train_or_test": "train",
    "on_hugging_face": True,
    "content_label_keys": (["sentence1", "sentence2"], "label"),
}

# MRPC
MRPC_DATASET_PARAMS = {
    "set_name": "glue",
    "config": "mrpc",
    "train_or_test": "train",
    "on_hugging_face": True,
    "content_label_keys": (["sentence1", "sentence2"], "label"),
}

# TWEET_EVAL_HATE
TWEET_EVAL_HATE_DATASET_PARAMS = {
    "set_name": "tweet_eval",
    "config": "hate",
    "train_or_test": "train",
    "on_hugging_face": True,
    "content_label_keys": (["text"], "label"),
}

# SICK
SICK_DATASET_PARAMS = {
    "set_name": "sick",
    "config": None,
    "train_or_test": "train",
    "on_hugging_face": True,
    "content_label_keys": (["sentence_A", "sentence_B"], "label"),
}

# POEM_SENTIMENT
POEM_SENTIMENT_DATASET_PARAMS = {
    "set_name": "poem_sentiment",
    "config": None,
    "train_or_test": "train",
    "on_hugging_face": True,
    "content_label_keys": (["verse_text"], "label"),
}

# ETHOS
ETHOS_DATASET_PARAMS = {
    "set_name": "ethos",
    "config": "binary",
    "train_or_test": "train",
    "on_hugging_face": True,
    "content_label_keys": (["text"], "label"),
}

# FINANCIAL_PHRASEBANK
FINANCIAL_PHRASEBANK_DATASET_PARAMS = {
    "set_name": "financial_phrasebank",
    "config": None,
    "train_or_test": "",
    "on_hugging_face": False,
    "content_label_keys": (["text"], "label"),
}

# MEDICAL_QUESTIONS_PAIRS
MEDICAL_QUESTIONS_PAIRS_DATASET_PARAMS = {
    "set_name": "medical_questions_pairs",
    "config": None,
    "train_or_test": "train",
    "on_hugging_face": True,
    "content_label_keys": (["question_1", "question_2"], "label"),
}

# TWEET_EVAL_STANCE_FEMINIST
TWEET_EVAL_STANCE_FEMINIST_DATASET_PARAMS = {
    "set_name": "tweet_eval",
    "config": "stance_feminist",
    "train_or_test": "train",
    "on_hugging_face": True,
    "content_label_keys": (["text"], "label"),
}

# TWEET_EVAL_STANCE_ATHEISM
TWEET_EVAL_STANCE_ATHEISM_DATASET_PARAMS = {
    "set_name": "tweet_eval",
    "config": "stance_atheism",
    "train_or_test": "train",
    "on_hugging_face": True,
    "content_label_keys": (["text"], "label"),
}

# UNNATURAL
UNNATURAL_DATASET_PARAMS = {
    "set_name": "unnatural",
    "config": None,
    "train_or_test": "",
    "on_hugging_face": False,
    "content_label_keys": (["text"], "label"),
}

# SST2_AB
SST2_AB_DATASET_PARAMS = {
    "set_name": "sst2",
    "config": None,
    "train_or_test": "train",
    "on_hugging_face": True,
    "content_label_keys": (["sentence"], "label"),
}

DATASET_PARAMS = {
    "sst2": SST2_DATASET_PARAMS,
    "agnews": AGNEWS_DATASET_PARAMS,
    "trec": TREC_DATASET_PARAMS,
    "dbpedia": DBPEDIA_DATASET_PARAMS,
    "rte": RTE_DATASET_PARAMS,
    "mrpc": MRPC_DATASET_PARAMS,
    "tweet_eval_hate": TWEET_EVAL_HATE_DATASET_PARAMS,
    "sick": SICK_DATASET_PARAMS,
    "poem_sentiment": POEM_SENTIMENT_DATASET_PARAMS,
    "ethos": ETHOS_DATASET_PARAMS,
    "financial_phrasebank": FINANCIAL_PHRASEBANK_DATASET_PARAMS,
    "medical_questions_pairs": MEDICAL_QUESTIONS_PAIRS_DATASET_PARAMS,
    "tweet_eval_stance_feminist": TWEET_EVAL_STANCE_FEMINIST_DATASET_PARAMS,
    "tweet_eval_stance_atheism": TWEET_EVAL_STANCE_ATHEISM_DATASET_PARAMS,
    "unnatural": UNNATURAL_DATASET_PARAMS,
    "sst2_ab": SST2_AB_DATASET_PARAMS,
}