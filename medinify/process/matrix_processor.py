from medinify.process import *
from torchtext.data import Field, LabelField, Example, BucketIterator
import torch
from torchtext.vocab import Vectors
from torchtext.data import Dataset as TorchtextDataset
from gensim.models import KeyedVectors
import numpy as np


class DataloaderProcessor(Processor):

    nickname = 'matrix'

    def __init__(self):
        super().__init__()
        self.w2v = None
        """
        self.text_field = Field(tokenize=super().tokenize, dtype=torch.float64)
        self.label_field = LabelField(dtype=torch.float64)
        """

    def get_features(self, dataset):
        dataset = remove_neutral(dataset)
        if not self.w2v:
            self.w2v = KeyedVectors.load_word2vec_format(dataset.args['word_embeddings'])
        comments = dataset.data_table[dataset.text_column]
        tokens = dataset.data_table.apply(lambda row: self.tokenize(row[dataset.text_column]), axis=1)
        print(tokens)
        exit()
        matrix = np.zeros((dataset.data_table[dataset.text_column].shape[0], 100))
        comments = dataset.data_table[dataset.text_column]
        for i, comment in enumerate(comments):
            tokens = self.tokenize(comment)
            all_embeddings = []
            for token in tokens:
                try:
                    all_embeddings.append(self.w2v[token])
                except KeyError:
                    continue
            if len(all_embeddings) == 0:
                continue
            else:
                embeddings[i] = np.average(all_embeddings, axis=0)
        """
        dataset = remove_neutral(dataset)
        self.get_labels(dataset)
        try:
            self.text_field.vocab
        except AttributeError:
            vectors = Vectors(dataset.args['word_embeddings'])
            self.text_field.build_vocab(dataset.data_table[dataset.text_column], vectors=vectors)
            self.label_field.build_vocab(dataset.data_table[dataset.label_column])

        fields = {'text': ('text', self.text_field), 'label': ('label', self.label_field)}
        text = dataset.data_table[dataset.text_column].to_numpy()
        labels = dataset.data_table['label'].to_numpy()
        examples = [Example.fromdict(
            data={'text': text[x], 'label': labels[x]}, fields=fields) for x in range(labels.shape[0])]
        dataset.data_table['matrices'] = examples
        return examples
        # torch_dataset = TorchtextDataset(examples, {'text': self.text_field, 'label': self.label_field})
        """

    def get_labels(self, dataset):
        dataset = remove_neutral(dataset)
        labels = BowProcessor().get_labels(dataset)
        return labels

    """
    @staticmethod
    def unpack_samples(loader):
        labels = []
        samples = []
        for sample in loader:
            labels.extend([x.item() for x in list(sample.label.to(torch.int64))])
            samples.append(sample)
        return labels, samples
    """

