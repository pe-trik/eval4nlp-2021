import os
import argparse
import torch
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

from transformers import Trainer, TrainingArguments
from transformers import XLMRobertaForTokenClassification, XLMRobertaTokenizerFast


MODEL_NAME = 'xlm-roberta-large'


class Dataset(torch.utils.data.Dataset):
    LABELS = [
        '0',
        '1'
    ]
    LABELS2ID = {
        '0': 0,
        '1': 1
    }

    def __init__(self, files, is_train=False):
        self.is_train = is_train
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(MODEL_NAME)

        self.encodings, self.labels = self._get_data(files)

    def _read_file(self, file):
        texts, labels = [], []
        srcs = open(f'{file}.src').readlines()
        mts = open(f'{file}.mt').readlines()
        stags = open(f'{file}.src-tags').readlines()
        ttags = open(f'{file}.tgt-tags').readlines()
        das = open(f'{file}.da').readlines()
        for src, mt, stag, ttag, da in zip(srcs, mts, stags, ttags, das):
            # concat string - extra ['</s>','</s>'] need to be placed in between
            text = src.strip().split(' ') + \
                ['</s>', '</s>'] + mt.strip().split(' ')
            # [-100,-100] are for the ['</s>','</s>']
            label = list(map(lambda x: int(x), stag.strip().split(
                ' '))) + [-100, -100] + list(map(lambda x: int(x), ttag.strip().split(' ')))
            texts.append(text)
            labels.append(label)
        return texts, labels

    def _get_data(self, files):
        texts, labels = [], []
        for file in files:
            t, l = self._read_file(file)
            texts += t
            labels += l
        encodings = self.tokenizer(texts, is_split_into_words=True,
                                   return_offsets_mapping=True, padding=True, truncation=False)
        encoded_labels = []
        for doc_labels, doc_offset, enc in zip(labels, encodings.offset_mapping, encodings['input_ids']):
            doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
            arr_offset = np.array(doc_offset)
            enc = np.array(enc)

            # set labels whose first offset position is 0 and the second is not 0
            #doc_enc_labels[0] = doc_labels[0]
            doc_enc_labels[(arr_offset[:, 0] == 0) & (
                arr_offset[:, 1] != 0)
                & (enc != 6)] = doc_labels  # it keeps creating SPACE token before punctuation symbols
            encoded_labels.append(doc_enc_labels.tolist())

        return encodings, encoded_labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        del item['offset_mapping']
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(p):
    def smax(x):
        m = np.max(x)
        return np.exp(x - m) / np.sum(np.exp(x - m))
    predictions, labels = p
    predictions = np.array([[smax(p) for p in sentence]
                           for sentence in predictions])
    y_score = np.array(predictions[:, :, 1])
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    das = [
        (prediction[0], label[0])
        for prediction, label in zip(predictions, labels)
    ]

    true_predictions = [
        [min(p, 1) for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    y_score = [
        [s for (p, l, s) in zip(prediction, label, score) if l != -100]
        for prediction, label, score in zip(predictions, labels, y_score)
    ]

    true_predictions = np.concatenate(true_predictions)
    true_labels = np.concatenate(true_labels)
    y_score = np.concatenate(y_score)

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
        "accuracy": accuracy_score(true_labels, true_predictions),
        "auc": roc_auc_score(true_labels, y_score)
    }


def main(args):
    dir_name = f'word-level_{MODEL_NAME}_{args.epochs}_{args.bs}_{args.lr}'
    dir_name = os.path.join(args.output, dir_name)

    model = XLMRobertaForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir=dir_name,          # output directory
        num_train_epochs=args.epochs,              # total number of training epochs
        per_device_train_batch_size=args.bs,  # batch size per device during training
        per_device_eval_batch_size=32,   # batch size for evaluation
        warmup_ratio=0.1,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,
        evaluation_strategy='epoch',              # strength of weight decay
        logging_steps=10,
        learning_rate=args.lr
    )

    train_dataset = Dataset(args.train, is_train=True)
    val_dataset = Dataset(args.val)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.evaluate()
    trainer.train()
    model.save_pretrained(dir_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--bs', type=int)
    parser.add_argument('--train', nargs='+')
    parser.add_argument('--val', nargs='+')
    parser.add_argument('--output', default='.')
    args = parser.parse_args()

    main(args)
