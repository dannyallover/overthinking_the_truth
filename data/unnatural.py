# Here we create the synthetic dataset, unnatural.
# source: https://ai.stanford.edu/blog/understanding-incontext/.

import random
import pandas as pd

DATA = {
    "plant/vegetable": [
        "onions",
        "broccoli",
        "kale",
        "beet",
        "corn",
        "beans",
        "zucchini",
        "celery",
        "lettuce",
        "radish",
        "cucumber",
        "peas",
        "tomato",
        "spinach",
        "carrots",
        "chard",
        "artichoke",
        "asparagus",
        "cauliflower",
        "cabbage",
        "endive",
        "squash",
        "sprout",
        "aubergine",
        "garlic",
        "dill",
        "parsley",
        "basil",
        "thyme",
        "leek",
    ],
    "sport": [
        "hockey",
        "golf",
        "football",
        "luge",
        "bowling",
        "archery",
        "lacrosse",
        "badminton",
        "billiards",
        "volleyball",
        "rugby",
        "cycling",
        "baseball",
        "tennis",
        "judo",
        "gymnastics",
        "marathon",
        "climbing",
        "triathlon",
        "swimming",
        "soccer",
        "diving",
        "shooting",
        "boxing",
        "rowing",
        "karate",
        "skiing",
        "snowboarding",
        "taekwondo",
        "weightlifting",
    ],
    "animal": [
        "horse",
        "sheep",
        "goldfish",
        "duck",
        "leopard",
        "lion",
        "porcupine",
        "wolf",
        "camel",
        "zebra",
        "llama",
        "cat",
        "elephant",
        "monkey",
        "panda",
        "tiger",
        "giraffe",
        "panda",
        "deer",
        "turtle",
        "cheetah",
        "sloth",
        "rhinoceros",
        "crocodile",
        "butterfly",
        "squirrel",
        "llama",
        "spider",
        "ant",
        "lizard",
    ],
}


def main():
    dataset = []
    for i, (k, v) in enumerate(DATA.items()):
        for v_ in v:
            dataset.append((v_, k))
    df = pd.DataFrame(dataset, columns=["text", "label"])
    df.to_csv("unnatural.csv")

if __name__ == "__main__":
    main()