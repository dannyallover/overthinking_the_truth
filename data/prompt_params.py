# SST2
SST2_LABELS = [
    ("Negative", "Positive"),
    ("bad", "good"),
    ("bad", "good"),
    ("negative", "positive"),
    ("bad", "good"),
    ("No", "Yes"),
    ("Negative", "Positive"),
    ("bad", "good"),
    ("bad", "good"),
    ("bad", "good"),
    ("liked", "hated"),
    ("0", "5"),
    ("Negative", "Positive"),
    ("False", "True"),
    ("0", "5"),
]
SST2_PROMPT_FORMATS = [
    "Review: {}\nAnswer: {}.",
    "Review: {}\nAnswer: {}.",
    "My review for last night’s film: {} The critics agreed that this movie was {}.",
    'One of our critics wrote "{}". Her sentiment towards the film was {}.',
    'In a contemporary review, Roger Ebert wrote: "{}". Entertainment Weekly  agreed, and the overall critical reception of the film was {}.',
    "Review: {}\nPositive Review? {}.",
    "Review: {}\nQuestion: Is the sentiment of the above review Positive or Negative?\nAnswer: {}.",
    "Review: {}\nQuestion: Did the author think that the movie was good or bad?\nAnswer: {}.",
    "Question: Did the author of the following tweet think that the movie was good or bad?\nTweet: {}\nAnswer: {}.",
    "{} My overall feeling was that the movie was {}.",
    "{} I {}.",
    "{} My friend asked me if I would give the movie 0 or 5 stars, I said {}.",
    "Input: {}\nSentiment: {}.",
    "Review: {}\nPositive: {}.",
    "Review: {}\nStars: {}.",
]
SST2_PREFIX_NARRATIVES = [
    "",
    "",
    "Here is what our critics think for this month’s films.",
    "Critical reception [ edit ].",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
]
SST2_PROMPT_PARAMS = {
    i: {
        "labels": SST2_LABELS[i],
        "prompt_format": SST2_PROMPT_FORMATS[i],
        "prefix_narrative": SST2_PREFIX_NARRATIVES[i],
    }
    for i in range(len(SST2_PROMPT_FORMATS))
}

# AGNEWS
AGNEWS_LABELS = [["World", "Sports", "Business", "Science"]]
AGNEWS_PROMPT_FORMATS = ["Article: {}\nAnswer: {}."]
AGNEWS_PREFIX_NARRATIVES = [""]
AGNEWS_PROMPT_PARAMS = {
    i: {
        "labels": AGNEWS_LABELS[i],
        "prompt_format": AGNEWS_PROMPT_FORMATS[i],
        "prefix_narrative": AGNEWS_PREFIX_NARRATIVES[i],
    }
    for i in range(len(AGNEWS_PROMPT_FORMATS))
}

# TREC
TREC_LABELS = [
    ["Description", "Entity", "Abbreviation", "Person", "Number", "Location"]
]
TREC_PROMPT_FORMATS = ["Question: {}\nAnswer Type: {}."]
TREC_PREFIX_NARRATIVES = [
    "Classify the questions based on whether their answer type is a Number, Location, Person, Description, Entity, or Abbreviation."
]
TREC_PROMPT_PARAMS = {
    i: {
        "labels": TREC_LABELS[i],
        "prompt_format": TREC_PROMPT_FORMATS[i],
        "prefix_narrative": TREC_PREFIX_NARRATIVES[i],
    }
    for i in range(len(TREC_PROMPT_FORMATS))
}

# DBPEDIA
DBPEDIA_LABELS = [
    [
        "Company",
        "School",
        "Artist",
        "Athlete",
        "Politician",
        "Transportation",
        "Building",
        "Nature",
        "Village",
        "Animal",
        "Plant",
        "Album",
        "Film",
        "Book",
    ]
]
DBPEDIA_PROMPT_FORMATS = ["Article: {}\nAnswer: {}."]
DBPEDIA_PREFIX_NARRATIVES = [
    "Classify the documents based on whether they are about a Company, School, Artist, Athlete, Politician, Transportation, Building, Nature, Village, Animal, Plant, Album, Film, or Book."
]
DBPEDIA_PROMPT_PARAMS = {
    i: {
        "labels": DBPEDIA_LABELS[i],
        "prompt_format": DBPEDIA_PROMPT_FORMATS[i],
        "prefix_narrative": DBPEDIA_PREFIX_NARRATIVES[i],
    }
    for i in range(len(DBPEDIA_PROMPT_FORMATS))
}

# RTE
RTE_LABELS = [["True", "False"]]
RTE_PROMPT_FORMATS = ["{}.\nquestion: {}. True or False?\nThe answer is: {}."]
RTE_PREFIX_NARRATIVES = [""]
RTE_PROMPT_PARAMS = {
    i: {
        "labels": RTE_LABELS[i],
        "prompt_format": RTE_PROMPT_FORMATS[i],
        "prefix_narrative": RTE_PREFIX_NARRATIVES[i],
    }
    for i in range(len(RTE_PROMPT_FORMATS))
}

# MRPC
MRPC_LABELS = [["False", "True"]]
MRPC_PROMPT_FORMATS = ["{}.\nquestion: {}. True or False?\nThe answer is: {}."]
MRPC_PREFIX_NARRATIVES = [""]
MRPC_PROMPT_PARAMS = {
    i: {
        "labels": MRPC_LABELS[i],
        "prompt_format": MRPC_PROMPT_FORMATS[i],
        "prefix_narrative": MRPC_PREFIX_NARRATIVES[i],
    }
    for i in range(len(MRPC_PROMPT_FORMATS))
}

# TWEET_EVAL_HATE
TWEET_EVAL_HATE_LABELS = [["favor", "against"]]
TWEET_EVAL_HATE_PROMPT_FORMATS = ["Tweet: {}\nSentiment: {}."]
TWEET_EVAL_HATE_PREFIX_NARRATIVES = [""]
TWEET_EVAL_HATE_PROMPT_PARAMS = {
    i: {
        "labels": TWEET_EVAL_HATE_LABELS[i],
        "prompt_format": TWEET_EVAL_HATE_PROMPT_FORMATS[i],
        "prefix_narrative": TWEET_EVAL_HATE_PREFIX_NARRATIVES[i],
    }
    for i in range(len(TWEET_EVAL_HATE_PROMPT_FORMATS))
}

# SICK
SICK_LABELS = [["True", "Not sure", "False"]]
SICK_PROMPT_FORMATS = ["{}.\nquestion: {}. True or False?\nThe answer is: {}."]
SICK_PREFIX_NARRATIVES = [""]
SICK_PROMPT_PARAMS = {
    i: {
        "labels": SICK_LABELS[i],
        "prompt_format": SICK_PROMPT_FORMATS[i],
        "prefix_narrative": SICK_PREFIX_NARRATIVES[i],
    }
    for i in range(len(SICK_PROMPT_FORMATS))
}

# POEM_SENTIMENT
POEM_SENTIMENT_LABELS = [["negative", "positive", "no impact"]]
POEM_SENTIMENT_PROMPT_FORMATS = ["{}:\nThe sentiment is: {}."]
POEM_SENTIMENT_PREFIX_NARRATIVES = [""]
POEM_SENTIMENT_PROMPT_PARAMS = {
    i: {
        "labels": POEM_SENTIMENT_LABELS[i],
        "prompt_format": POEM_SENTIMENT_PROMPT_FORMATS[i],
        "prefix_narrative": POEM_SENTIMENT_PREFIX_NARRATIVES[i],
    }
    for i in range(len(POEM_SENTIMENT_PROMPT_FORMATS))
}

# ETHOS
ETHOS_LABELS = [["no", "yes"]]
ETHOS_PROMPT_FORMATS = ["Text: {}\nAnswer: {}."]
ETHOS_PREFIX_NARRATIVES = ["Is the following hate speech? Answer yes or no."]
ETHOS_PROMPT_PARAMS = {
    i: {
        "labels": ETHOS_LABELS[i],
        "prompt_format": ETHOS_PROMPT_FORMATS[i],
        "prefix_narrative": ETHOS_PREFIX_NARRATIVES[i],
    }
    for i in range(len(ETHOS_PROMPT_FORMATS))
}

# FINANCIAL_PHRASEBANK
FINANCIAL_PHRASEBANK_LABELS = [["Negative", "Neutral", "Positive"]]
FINANCIAL_PHRASEBANK_PROMPT_FORMATS = ["Text: {}\nSentiment: {}."]
FINANCIAL_PHRASEBANK_PREFIX_NARRATIVES = [""]
FINANCIAL_PHRASEBANK_PROMPT_PARAMS = {
    i: {
        "labels": FINANCIAL_PHRASEBANK_LABELS[i],
        "prompt_format": FINANCIAL_PHRASEBANK_PROMPT_FORMATS[i],
        "prefix_narrative": FINANCIAL_PHRASEBANK_PREFIX_NARRATIVES[i],
    }
    for i in range(len(FINANCIAL_PHRASEBANK_PROMPT_FORMATS))
}

# MEDICAL_QUESTIONS_PAIRS
MEDICAL_QUESTIONS_PAIRS_LABELS = [("not", "equivalent")]
MEDICAL_QUESTIONS_PAIRS_PROMPT_FORMATS = ["Question: {}.\nQuestion: {}.\nAnswer: {}."]
MEDICAL_QUESTIONS_PAIRS_PREFIX_NARRATIVES = [
    "Determine if the two questions are equivalent or not."
]
MEDICAL_QUESTIONS_PAIRS_PROMPT_PARAMS = {
    i: {
        "labels": MEDICAL_QUESTIONS_PAIRS_LABELS[i],
        "prompt_format": MEDICAL_QUESTIONS_PAIRS_PROMPT_FORMATS[i],
        "prefix_narrative": MEDICAL_QUESTIONS_PAIRS_PREFIX_NARRATIVES[i],
    }
    for i in range(len(MEDICAL_QUESTIONS_PAIRS_PROMPT_FORMATS))
}

# TWEET_EVAL_STANCE_FEMINIST
TWEET_EVAL_STANCE_FEMINIST_LABELS = [["neither", "no", "yes"]]
TWEET_EVAL_STANCE_FEMINIST_PROMPT_FORMATS = ["Tweet: {}\nAnswer: {}."]
TWEET_EVAL_STANCE_FEMINIST_PREFIX_NARRATIVES = [
    "Determine if the text supports feminism. Answer with yes, no, or neither."
]
TWEET_EVAL_STANCE_FEMINIST_PROMPT_PARAMS = {
    i: {
        "labels": TWEET_EVAL_STANCE_FEMINIST_LABELS[i],
        "prompt_format": TWEET_EVAL_STANCE_FEMINIST_PROMPT_FORMATS[i],
        "prefix_narrative": TWEET_EVAL_STANCE_FEMINIST_PREFIX_NARRATIVES[i],
    }
    for i in range(len(TWEET_EVAL_STANCE_FEMINIST_PROMPT_FORMATS))
}

# TWEET_EVAL_STANCE_ATHEISM
TWEET_EVAL_STANCE_ATHEISM_LABELS = [["neither", "no", "yes"]]
TWEET_EVAL_STANCE_ATHEISM_PROMPT_FORMATS = ["Tweet: {}\nAnswer: {}."]
TWEET_EVAL_STANCE_ATHEISM_PREFIX_NARRATIVES = [
    "Determine if the text supports atheism. Answer with yes, no, or neither."
]
TWEET_EVAL_STANCE_ATHEISM_PROMPT_PARAMS = {
    i: {
        "labels": TWEET_EVAL_STANCE_ATHEISM_LABELS[i],
        "prompt_format": TWEET_EVAL_STANCE_ATHEISM_PROMPT_FORMATS[i],
        "prefix_narrative": TWEET_EVAL_STANCE_ATHEISM_PREFIX_NARRATIVES[i],
    }
    for i in range(len(TWEET_EVAL_STANCE_ATHEISM_PROMPT_FORMATS))
}

# UNNATURAL
UNNATURAL_LABELS = [["animal", "plant/vegetable", "sport"]]
UNNATURAL_PROMPT_FORMATS = ["{}: {}."]
UNNATURAL_PREFIX_NARRATIVES = [
    "Consider the categories plant/vegetable, sport, and animal. Classify each object in its category."
]
UNNATURAL_PROMPT_PARAMS = {
    i: {
        "labels": UNNATURAL_LABELS[i],
        "prompt_format": UNNATURAL_PROMPT_FORMATS[i],
        "prefix_narrative": UNNATURAL_PREFIX_NARRATIVES[i],
    }
    for i in range(len(UNNATURAL_PROMPT_FORMATS))
}

# SST2_AB
SST2_AB_LABELS = [("A", "B")]
SST2_AB_PROMPT_FORMATS = ["Review: {}\nAnswer: {}."]
SST2_AB_PREFIX_NARRATIVES = [""]
SST2_AB_PROMPT_PARAMS = {
    i: {
        "labels": SST2_AB_LABELS[i],
        "prompt_format": SST2_AB_PROMPT_FORMATS[i],
        "prefix_narrative": SST2_AB_PREFIX_NARRATIVES[i],
    }
    for i in range(len(SST2_AB_PROMPT_FORMATS))
}


PROMPT_PARAMS = {
    "sst2": SST2_PROMPT_PARAMS,
    "agnews": AGNEWS_PROMPT_PARAMS,
    "trec": TREC_PROMPT_PARAMS,
    "dbpedia": DBPEDIA_PROMPT_PARAMS,
    "rte": RTE_PROMPT_PARAMS,
    "mrpc": MRPC_PROMPT_PARAMS,
    "tweet_eval_hate": TWEET_EVAL_HATE_PROMPT_PARAMS,
    "sick": SICK_PROMPT_PARAMS,
    "poem_sentiment": POEM_SENTIMENT_PROMPT_PARAMS,
    "ethos": ETHOS_PROMPT_PARAMS,
    "financial_phrasebank": FINANCIAL_PHRASEBANK_PROMPT_PARAMS,
    "medical_questions_pairs": MEDICAL_QUESTIONS_PAIRS_PROMPT_PARAMS,
    "tweet_eval_stance_feminist": TWEET_EVAL_STANCE_FEMINIST_PROMPT_PARAMS,
    "tweet_eval_stance_atheism": TWEET_EVAL_STANCE_ATHEISM_PROMPT_PARAMS,
    "unnatural": UNNATURAL_PROMPT_PARAMS,
    "sst2_ab": SST2_AB_PROMPT_PARAMS,
}