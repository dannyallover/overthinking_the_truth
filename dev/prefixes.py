import datasets
import random
import pandas as pd
from transformers import AutoTokenizer

datasets.disable_progress_bar()

def get_dataset(dataset_params: dict) -> list:
    """
    Get the dataset based off |dataset_params|.

    Parameters
    ----------
    dataset_params : required, dict
        Dataset metadata.

    Returns
    ------
    dataset : list
        List of examples for each class in the dataset.
    """
    set_name, config, train_or_test, on_hugging_face, content_label_keys = (
        dataset_params["set_name"],
        dataset_params["config"],
        dataset_params["train_or_test"],
        dataset_params["on_hugging_face"],
        dataset_params["content_label_keys"],
    )
    if on_hugging_face:
        raw_data_df = datasets.load_dataset(set_name, config)[train_or_test].to_pandas()
    else:
        raw_data_df = pd.read_csv(f"../../data/{set_name}.csv")

    content_keys, label_key = content_label_keys
    labels = raw_data_df[label_key].unique()
    labels.sort()

    dataset = []
    for l in labels:
        chained_content = [
            raw_data_df[raw_data_df[label_key] == l][content_keys[i]].values
            for i in range(len(content_keys))
        ]
        interleaved_content = list(zip(*chained_content))
        dataset.append(interleaved_content)

    return dataset

class Prefixes:
    """
    Class to build the prefixes.

    Attributes
    ----------
    true_prefixes : list
        |num_inputs| true prefixes, each containing
        |num_demos| demonstrations.
    false_prefixes : list
        |num_inputs| false prefixes prescribed by |demo_params|, each containing
        |num_demos| demonstrations.
    prefixes_true_labels : list
        Correct label corresponding to each position in context and prefix.
    prefixes_false_labels : list
        Labels used for false demonstrations for each position in context and
        prefix.
    true_prefixes_tok : list
        Tokenized |true_prefixes|.
    false_prefixes_tok : list
        Tokenized |false_prefixes|.
    true_prefixes_tok_prec_label_indx : list
        Indices of the token preceding the labels in the demonstrations for
        true prefixes.
    false_prefixes_tok_prec_label_indx : list
        Indices of the token preceding the labels in the demonstrations for
        false prefixes.
    true_prefixes_tok_label_indx : list
        Indices of the labels in the demonstrations for true prefixes.
    false_prefixes_tok_label_indx : list
        Indices of the labels in the demonstrations for false prefixes.
    """

    def __init__(
        self,
        dataset: list,
        prompt_params: dict,
        demo_params: dict,
        model_params: dict,
        tokenizer: AutoTokenizer,
        num_inputs: int,
        num_demos: int,
    ):
        """
        Initializes class.

        Parameters
        ----------
        dataset : required, list
            List of examples for each class in the dataset.
        prompt_params : required, dict
            Prompt metadata.
        demo_params : required, dict
            Demo metadata.
        model_params : required, dict
            Model metadata.
        tokenizer : required, AutoTokenizer
            Tokenizer.
        num_inputs : required, int
            The number of inputs for each prefix type.
        num_inputs : required, int
            The number of demos for each input.

        Returns
        ------
        None
        """
        self.true_prefixes = []
        self.false_prefixes = []
        self.true_prefixes_labels = []
        self.false_prefixes_labels = []
        self.true_prefixes_tok = []
        self.false_prefixes_tok = []
        self.true_prefixes_tok_prec_label_indx = []
        self.true_prefixes_tok_label_indx = []
        self.false_prefixes_tok_prec_label_indx = []
        self.false_prefixes_tok_label_indx = []

        self.__set_prefixes_and_labels(
            dataset,
            prompt_params,
            demo_params,
            model_params,
            tokenizer,
            num_inputs,
            num_demos,
        )

    def __get_tok_labels_indx(
        self,
        demos: list,
        prefix_narrative: str,
        labels: list,
        label_strs: list,
        tokenizer: AutoTokenizer,
    ) -> list:
        """
        Finds the positions of the labels and the tokens preceding the labels.

        Parameters
        ----------
        demos : required, list
            Tuple of strings that comprise the sample.
        prefix_narrative : required, str
            The string that precedes the demonstrations in an input.
        labels : required, list
            List of the label indices for each position in context.
        label_strs : required, list
            List of all labels as strings.
        tokenizer : required, AutoTokenizer
            Tokenizer.

        Returns
        ------
        None
        """
        PNLNL = len(tokenizer(".\n\n")["input_ids"])
        running_count = (
            len(tokenizer(prefix_narrative)["input_ids"]) + PNLNL
            if prefix_narrative
            else 0
        )
        prec_lab_indices, lab_indices = [], []
        for i in range(len(demos)):
            len_demo = len(tokenizer(demos[i])["input_ids"])
            len_label = len(tokenizer(" " + label_strs[labels[i]])["input_ids"])
            indx = running_count + len_demo - len_label - PNLNL
            running_count += len_demo + PNLNL
            prec_lab_indices.append(indx)
            lab_indices.append([i for i in range(indx, indx + len_label + 1)])
        return prec_lab_indices, lab_indices

    def __get_sample_len(self, sample: tuple, tokenizer: AutoTokenizer):
        """
        Gets the number of tokens in |sample|.

        Parameters
        ----------
        sample : required, tuple
            Tuple of strings that comprise the sample.
        tokenizer : required,AutoTokenizer
            Tokenizer.

        Returns
        ------
        None
        """
        sample_len = 0
        for samp in sample:
            sample_len += len(tokenizer(samp)["input_ids"])
        return sample_len

    def __set_prefixes_and_labels(
        self,
        dataset: list,
        prompt_params: dict,
        demo_params: dict,
        model_params: dict,
        tokenizer: AutoTokenizer,
        num_inputs: int,
        num_demos: int,
    ) -> None:
        """
        Sets |self.true_prefixes| and |self.false_prefixes| to the built true
        and false prefixes, and |self.prefixes_true_labels| and |self.prefixes_false_labels|
        to the labels in the true and false prefixes at each position in context. Saves
        the positions of the labels and the tokens preceding the labels.

        Parameters
        ----------
        dataset : required, list
            List of examples for each class in the dataset.
        prompt_params : required, dict
            Prompt metadata.
        demo_params : required, dict
            Demo metadata.
        model_params : required, dict
            Model metadata.
        tokenizer : required, AutoTokenizer
            Tokenizer.
        num_inputs : required, int
            The number of inputs for each prefix type.
        num_inputs : required, int
            The number of demos for each input.

        Returns
        ------
        None
        """
        prefix_narrative = prompt_params["prefix_narrative"]
        labels = prompt_params["labels"]
        prompt_format = prompt_params["prompt_format"]
        max_token_len = model_params["max_token_len"]
        PNLNL = len(tokenizer(".\n\n")["input_ids"])

        avg_samp_size = (
            (max_token_len // num_demos)
            - max([len(tokenizer(lab)["input_ids"]) for lab in labels])
            - len(tokenizer(prompt_format)["input_ids"])
            - (len(tokenizer(prefix_narrative)["input_ids"]) // num_demos)
            - PNLNL
        )

        i = 0
        while i < num_inputs:
            true_demos, false_demos = [], []
            true_labels, false_labels = [], []
            cycle_start = random.randrange(1, len(labels))
            samples = set()
            curr_len = 0
            j = 0
            while j < num_demos:
                true_lab = random.randrange(0, len(labels))
                incorrect_labs = list(range(0, true_lab)) + list(
                    range(true_lab + 1, len(labels))
                )

                sample = random.sample(dataset[true_lab], 1)[0]
                len_sample = self.__get_sample_len(sample, tokenizer)
                curr_avg = (curr_len + len_sample) / (j + 1)

                while sample in samples or curr_avg > avg_samp_size:
                    sample = random.sample(dataset[true_lab], 1)[0]
                    len_sample = self.__get_sample_len(sample, tokenizer)
                    curr_avg = (curr_len + len_sample) / (j + 1)
                samples.add(sample)
                curr_len += len_sample + PNLNL

                if demo_params["percent_true"] >= random.uniform(0, 1):
                    false_lab = true_lab  # change name
                elif demo_params["permuted_incorrect"]:
                    false_lab = (true_lab + cycle_start) % len(labels)
                elif demo_params["random_incorrect"]:
                    false_lab = random.choice(
                        incorrect_labs
                    )  # notably choose amongst incorrect labels
                elif demo_params["random"]:
                    false_lab = random.randrange(0, len(labels))

                true_demo = prompt_format.format(*sample, labels[true_lab])
                false_demo = prompt_format.format(*sample, labels[false_lab])
                true_demos.append(true_demo)
                false_demos.append(false_demo)
                true_labels.append(true_lab)
                false_labels.append(false_lab)
                j += 1

            combined = list(zip(true_demos, false_demos, true_labels, false_labels))
            random.shuffle(combined)
            true_demos, false_demos, true_labels, false_labels = zip(*combined)

            prec_lab_indices, lab_indices = self.__get_tok_labels_indx(
                true_demos, prefix_narrative, true_labels, labels, tokenizer
            )
            self.true_prefixes_tok_prec_label_indx.append(prec_lab_indices)
            self.true_prefixes_tok_label_indx.append(lab_indices)
            prec_lab_indices, lab_indices = self.__get_tok_labels_indx(
                false_demos, prefix_narrative, false_labels, labels, tokenizer
            )
            self.false_prefixes_tok_prec_label_indx.append(prec_lab_indices)
            self.false_prefixes_tok_label_indx.append(lab_indices)

            if prefix_narrative:
                true_demos = (prefix_narrative,) + true_demos
                false_demos = (prefix_narrative,) + false_demos

            true_prefix = "\n\n".join(true_demos)
            false_prefix = "\n\n".join(false_demos)
            true_prefix_t = tokenizer(
                true_prefix,
                return_tensors="pt",
                padding=True,
            )
            false_prefix_t = tokenizer(
                false_prefix,
                return_tensors="pt",
                padding=True,
            )

            self.true_prefixes.append(true_prefix)
            self.false_prefixes.append(false_prefix)
            self.true_prefixes_labels.append(true_labels)
            self.false_prefixes_labels.append(false_labels)
            self.true_prefixes_tok.append(true_prefix_t)
            self.false_prefixes_tok.append(false_prefix_t)
            i += 1

        self.lab_first_token_ids = [
            token[0] for token in tokenizer([" " + lab for lab in labels])["input_ids"]
        ]
        self.num_labels = len(labels)

        return None