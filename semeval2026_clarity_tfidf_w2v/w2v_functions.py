import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec


class BaseW2VAnalyzer:
    def __init__(self, text_column="clean_text", pooling="mean", lowercase=False):
        """
        Base class for analyzing text with Word2Vec embeddings.

        Parameters:
            text_column (str): Name of the column that contains the text.
            pooling (str): Pooling strategy to combine word vectors into a document vector.
                           Supported values: "mean", "max", "mean_max", "tfidf_weighted_mean".
            lowercase (bool): Whether to convert text to lowercase before tokenization.
        """
        self.text_column = text_column
        self.pooling = pooling
        self.lowercase = lowercase

        self.wv = None
        self.vector_size = None  # Will be defined by the child classes
        self.tfidf_vectorizer = None
        self.idf_dict = None

        valid_poolings = {"mean", "max", "mean_max", "tfidf_weighted_mean"}
        if self.pooling not in valid_poolings:
            raise ValueError(f"pooling must be one of {valid_poolings}")

    def tokenize_text(self, df):
        """
        Tokenizes the text column of a DataFrame.

        Parameters:
            df (pd.DataFrame): Input DataFrame containing the text column.

        Returns:
            pd.Series: A series of tokenized texts.
        """
        text_series = df[self.text_column].fillna("").astype(str)
        if self.lowercase:
            text_series = text_series.str.lower()
        return text_series.apply(lambda x: x.split())

    def _fit_tfidf(self, df):
        """
        Fits a TF-IDF vectorizer only when tfidf_weighted_mean pooling is used.
        Stores the IDF values in a dictionary for later weighting.
        """
        if self.pooling == "tfidf_weighted_mean":
            self.tfidf_vectorizer = TfidfVectorizer(lowercase=self.lowercase)
            texts = df[self.text_column].fillna("").astype(str)
            self.tfidf_vectorizer.fit(texts)
            self.idf_dict = dict(
                zip(
                    self.tfidf_vectorizer.get_feature_names_out(),
                    self.tfidf_vectorizer.idf_
                )
            )

    def document_vector(self, tokens):
        """
        Converts a tokenized document into a single vector using the selected pooling strategy.

        Parameters:
            tokens (list): List of tokens from a document.

        Returns:
            np.ndarray: Document embedding vector.
        """
        # Use self.wv, which is available in both pretrained and newly trained models
        valid_tokens = [w for w in tokens if w in self.wv]

        if not valid_tokens:
            size = self.vector_size * 2 if self.pooling == "mean_max" else self.vector_size
            return np.zeros(size)

        if self.pooling == "tfidf_weighted_mean":
            counts = Counter(valid_tokens)
            weighted_vectors = []
            weights = []

            for word in set(valid_tokens):
                tf = counts[word]
                idf = self.idf_dict.get(word, 1.0)
                weighted_vectors.append(self.wv[word])
                weights.append(tf * idf)

            return np.average(weighted_vectors, axis=0, weights=weights)

        # Shared logic for mean, max, and mean_max pooling
        vectors = np.array([self.wv[w] for w in valid_tokens])

        if self.pooling == "mean":
            return np.mean(vectors, axis=0)
        elif self.pooling == "max":
            return np.max(vectors, axis=0)
        elif self.pooling == "mean_max":
            return np.concatenate([np.mean(vectors, axis=0), np.max(vectors, axis=0)])

    def transform(self, df):
        """
        Transforms a DataFrame of text documents into document vectors.

        Parameters:
            df (pd.DataFrame): Input DataFrame containing text data.

        Returns:
            np.ndarray: Matrix of document vectors.
        """
        if self.wv is None:
            raise ValueError("Model vectors are not loaded or trained yet.")

        tokens_series = self.tokenize_text(df)
        X = np.vstack([self.document_vector(tokens) for tokens in tokens_series])
        return X

    def fit_transform(self, df):
        """
        Fits the model and transforms the input DataFrame into document vectors.

        Parameters:
            df (pd.DataFrame): Input DataFrame containing text data.

        Returns:
            np.ndarray: Matrix of document vectors.
        """
        self.fit(df)
        return self.transform(df)


# ==========================================
# 2. Class for the Pretrained Model
# ==========================================
class PretrainedWord2VecAnalyzer(BaseW2VAnalyzer):
    def __init__(self, model_path=None, model=None, **kwargs):
        """
        Analyzer for a pretrained Word2Vec model.

        Parameters:
            model_path (str): Path to a saved KeyedVectors model.
            model: Already loaded pretrained model object.
            **kwargs: Additional arguments passed to BaseW2VAnalyzer.
        """
        super().__init__(**kwargs)
        self.model_path = model_path

        # Load directly if a model object is passed, otherwise load from path
        if model is not None:
            self.wv = model
            self.vector_size = self.wv.vector_size
        elif self.model_path is not None:
            self.load_model()
        else:
            raise ValueError("Provide either 'model_path' or 'model'.")

    def load_model(self):
        """
        Loads pretrained word vectors from disk.
        """
        from gensim.models import KeyedVectors
        self.wv = KeyedVectors.load(self.model_path)
        self.vector_size = self.wv.vector_size

    def fit(self, df):
        """
        Fits any additional components needed for the analyzer.
        For pretrained Word2Vec, this only applies TF-IDF fitting when needed.

        Parameters:
            df (pd.DataFrame): Input DataFrame containing text data.

        Returns:
            self
        """
        self._fit_tfidf(df)
        return self


# ==========================================
# 3. Class for Training a New Model
# ==========================================
class Word2VecAnalyzer(BaseW2VAnalyzer):
    def __init__(self, vector_size=400, window=3, min_count=3, workers=4, sg=1, epochs=20, **kwargs):
        """
        Analyzer that trains a new Word2Vec model from the given dataset.

        Parameters:
            vector_size (int): Dimensionality of the word vectors.
            window (int): Maximum distance between the current and predicted word.
            min_count (int): Ignores words with total frequency lower than this.
            workers (int): Number of worker threads.
            sg (int): Training algorithm: 1 for Skip-Gram, 0 for CBOW.
            epochs (int): Number of training iterations.
            **kwargs: Additional arguments passed to BaseW2VAnalyzer.
        """
        super().__init__(**kwargs)
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.epochs = epochs
        self.model = None

    def fit(self, df):
        """
        Trains a new Word2Vec model on the input text data.

        Parameters:
            df (pd.DataFrame): Input DataFrame containing text data.

        Returns:
            self
        """
        tokens_series = self.tokenize_text(df)

        self.model = Word2Vec(
            sentences=tokens_series,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg,
            epochs=self.epochs
        )

        # Store the learned word vectors in self.wv
        # so that BaseW2VAnalyzer can use them
        self.wv = self.model.wv

        self._fit_tfidf(df)
        return self