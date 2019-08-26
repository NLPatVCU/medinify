
import spacy
import ast
import torch
import pandas as pd
from nltk.corpus import stopwords
import tempfile
from medinify import config
from torchtext.data import Field, LabelField, TabularDataset, BucketIterator
from torchtext.vocab import Vectors


class CNNDataset:

    COMMENT = None

    def __init__(self, rating_type=None):
        self.nlp = spacy.load('en_core_web_sm')
        self.stops = stopwords.words('english')
        if not config.RATING_TYPE:
            config.RATING_TYPE = rating_type
        self.COMMENT = Field(tokenize=self.tokenize, dtype=torch.float64)
        self.RATING = LabelField(preprocessing=_preprocess_rating, dtype=torch.float64)

    def get_dataset(self, review_file=None, comments=None, ratings=None):
        if review_file:
            dataset = TabularDataset(
                path=review_file,
                format='csv',
                fields=[('comment', self.COMMENT), ('rating', self.RATING)],
                skip_header=True
            )

        else:
            reviews_dict = {'comments': comments, 'ratings': ratings}
            reviews_df = pd.DataFrame(reviews_dict)

            with tempfile.NamedTemporaryFile() as temp:
                reviews_df.to_csv(temp.name, index=False)
                dataset = TabularDataset(
                    path=temp.name,
                    format='csv',
                    fields=[('comment', self.COMMENT), ('rating', self.RATING)],
                    skip_header=True
                )

        dataset.examples = [example for example in dataset.examples if example.rating != 'neutral']
        return dataset

    def _build_vocabs(self, datasets, w2v_file):
        vectors = Vectors(w2v_file)
        self.COMMENT.build_vocab(*datasets, vectors=vectors)
        self.RATING.build_vocab(*datasets)

    def get_data_loader(self, w2v_file, review_file=None, comments=None, ratings=None):
        dataset = [self.get_dataset(review_file=review_file, comments=comments, ratings=ratings)]
        self._build_vocabs(dataset, w2v_file)
        loader = BucketIterator(dataset=dataset[0], batch_size=config.BATCH_SIZE)
        return loader

    def get_data_loaders(self, w2v_file, train_file=None, validation_file=None,
                         train_comments=None, train_rating=None,
                         validation_comment=None, validation_ratings=None):
        train_dataset = self.get_dataset(comments=train_comments,
                                         ratings=train_rating,
                                         review_file=train_file)
        validation_dataset = self.get_dataset(comments=validation_comment,
                                              ratings=validation_ratings,
                                              review_file=validation_file)
        datasets = [train_dataset, validation_dataset]

        self._build_vocabs(datasets, w2v_file)

        train_loader = BucketIterator(train_dataset, config.BATCH_SIZE)
        validation_loader = BucketIterator(validation_dataset, config.BATCH_SIZE)
        return train_loader, validation_loader

    def tokenize(self, comment):
        tokens = [token.text for token in self.nlp.tokenizer(comment.lower())
                  if not token.is_punct and not token.is_space and token.text not in self.stops]
        return tokens


def _preprocess_rating(rating):
    ratings_dict = ast.literal_eval(rating)
    rating = int(ratings_dict[config.RATING_TYPE])
    if rating == 3:
        rating = 'neutral'
    elif rating in [4, 5]:
        rating = 'pos'
    elif rating in [1, 2]:
        rating = 'neg'
    return rating
