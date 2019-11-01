from medinify.process import *
from torchtext.data import Field, LabelField, Example, BucketIterator
import torch
from torchtext.vocab import Vectors
from torchtext.data import Dataset as TorchtextDataset


class DataloderProcessor(Processor):

    def __init__(self):
        super().__init__()
        self.text_field = Field(tokenize=super().tokenize, dtype=torch.float64)
        self.label_field = LabelField(dtype=torch.float64)

    def get_features(self, dataset):
        dataset = remove_neutral(dataset)
        vectors = Vectors(dataset.word_embeddings)
        self.get_labels(dataset)
        fields = {'text': ('text', self.text_field), 'label': ('label', self.label_field)}
        text = dataset.data_table[dataset.text_column].to_numpy()
        labels = dataset.data_table['label'].to_numpy()
        examples = [Example.fromdict(
            data={'text': text[x], 'label': labels[x]}, fields=fields) for x in range(labels.shape[0])]
        torch_dataset = TorchtextDataset(examples, {'text': self.text_field, 'label': self.label_field})
        try:
            self.text_field.vocab
        except AttributeError:
            self.text_field.build_vocab(torch_dataset, vectors=vectors)
            self.label_field.build_vocab(torch_dataset)
        loader = BucketIterator(torch_dataset, batch_size=25)
        return loader

    def get_labels(self, dataset):
        dataset = remove_neutral(dataset)
        labels = BowProcessor().get_labels(dataset)
        return labels

