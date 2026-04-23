# Libraries
from w2v_functions import *
from tfidf_functions import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from IPython.display import display

# This function displays random samples from the DataFrame in a clear format,
# showing all columns for each sample.
def display_samples(df, num_samples=2, random_state=None):
    pd.set_option("display.max_colwidth", None)

    samples = df.sample(n=num_samples, random_state=random_state)

    for i, row in samples.iterrows():
        print(f"Sample {i}")

        sample_table = row.to_frame(name="Value")
        display(sample_table)


# This function creates pie charts for the distribution of clarity labels for each president,
# as well as a bar chart comparing the label distribution across presidents.
def plot_clarity_distribution_by_president(df):
    table = pd.crosstab(df["president"], df["clarity_label"])
    colors = ["#1f77b4", "#a99ad3", "#b8709a"]

    for president in table.index:
        table.loc[president].plot(
            kind="pie",
            autopct="%1.1f%%",
            startangle=90,
            figsize=(5, 5),
            colors=colors
        )

        plt.title(f"Clarity Labels for {president}")
        plt.ylabel("")
        plt.show()

    label_by_president = df.groupby(["president", "clarity_label"]).size().unstack()

    label_by_president.plot(
        kind="bar",
        stacked=False,
        figsize=(10, 6),
        color=colors
    )
    plt.title("Label Distribution by President")
    plt.xlabel("President")
    plt.ylabel("Number of Instances")
    plt.legend(title="Clarity Label")
    plt.xticks(rotation=45)

    plt.show()


def count_words(df, column_name="interview_answer", new_col="answer_word_count", label_col=None, bins=30):
    """
    Counts the number of words in a text column and displays summary statistics
    and histograms for the entire dataset and optionally by label.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Name of the text column to analyze.
        new_col (str): Name of the new column that will store word counts.
        label_col (str, optional): Label column for grouped analysis.
        bins (int): Number of bins for the histograms.
    """
    df = df.copy()

    # Compute the word count for each row
    df[new_col] = df[column_name].astype(str).str.split().str.len()

    # Overall statistics
    print("=== Overall Statistics ===")
    print(df[new_col].describe())
    print("Shortest answer:", df[new_col].min())
    print("Longest answer:", df[new_col].max())

    # Histogram for the entire dataset
    plt.figure(figsize=(8, 5))
    plt.hist(df[new_col], bins=bins, edgecolor="black")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.title(f"Length Distribution for: {column_name}")
    plt.show()

    # If a label column is provided, perform grouped analysis by label
    if label_col is not None:
        print(f"\n=== Statistics by {label_col} ===")
        print(df.groupby(label_col)[new_col].agg(["count", "mean", "median", "min", "max", "std"]))

        # Separate histogram for each label
        for label in df[label_col].dropna().unique():
            subset = df[df[label_col] == label]

            plt.figure(figsize=(8, 5))
            plt.hist(subset[new_col], bins=bins, edgecolor="black")
            plt.xlabel("Number of Words")
            plt.ylabel("Frequency")
            plt.title(f"Length Distribution for {column_name} - {label_col}={label}")
            plt.show()


class ExperimentTracker:
    def __init__(self, experiment_name, model_search_func, vectorizer, clean_func):
        """
        Initializes the experiment tracker.

        Args:
            experiment_name (str): Identifier for the experiment.
            model_search_func (callable): Function that handles grid search or model training.
            vectorizer (object): Text vectorizer (e.g., TFIDFProcessor or Word2VecAnalyzer).
            clean_func (callable): Function used to clean the raw text.
        """
        self.experiment_name = experiment_name
        self.model_search_func = model_search_func
        self.vectorizer = vectorizer
        self.clean_func = clean_func

        # Tracking variables
        self.best_model = None
        self.evaluator = None
        self.X_train = None
        self.y_train = None

    def run_training(self, training_data, text_col, label_col, clean_params):
        """
        Executes the pipeline:
        Preprocessing -> Vectorization -> Training -> Evaluation.
        """
        print(f"--- Starting Experiment: {self.experiment_name} ---")

        # 1. Text cleaning
        print("1. Cleaning text...")
        clean_text_series = training_data[text_col].apply(
            lambda x: self.clean_func(x, **clean_params)
        )

        # 2. Vectorization
        print(f"2. Vectorization using {self.vectorizer.__class__.__name__}...")

        # Smart input selection:
        # DataFrame for Word2Vec-based classes, Series for traditional TF-IDF
        if hasattr(self.vectorizer, "text_column"):
            train_input = pd.DataFrame({text_col: clean_text_series})
        else:
            train_input = clean_text_series

        self.X_train = self.vectorizer.fit_transform(train_input)
        self.y_train = training_data[label_col]

        if hasattr(self.vectorizer, "print_summary"):
            self.vectorizer.print_summary(self.X_train)

        # 3. Model training
        print("\n3. Training Model & Hyperparameter Tuning...")
        self.best_model = self.model_search_func(self.X_train, self.y_train)

        # 4. Cross-validation evaluation
        print("\n4. Cross-Validation Evaluation...")
        self.evaluator = ModelEvaluator(model=self.best_model, cv_folds=5)
        self.evaluator.run_evaluation(self.X_train, self.y_train, print_report=True)

        print(f"\nExperiment '{self.experiment_name}' completed and saved.")

    def evaluate_on_test(self, test_data, text_col, label_col, clean_params=None):
        """
        Evaluates the trained pipeline on the provided test set.

        Args:
            test_data (pd.DataFrame): Test dataset.
            text_col (str): Name of the text column.
            label_col (str): Name of the target label column.
            clean_params (dict, optional): Parameters for the cleaning function.

        Returns:
            tuple: True labels and predicted labels for the test set.
        """
        if self.best_model is None:
            print("Error: Model has not been trained yet.")
            return

        print(f"\n{'=' * 70}")
        print(f"TEST SET Evaluation for experiment: {self.experiment_name}")
        print(f"{'=' * 70}")

        # 1. Clean the test set
        params = clean_params if clean_params is not None else {}
        print("Cleaning test set text...")
        clean_test_series = test_data[text_col].apply(
            lambda x: self.clean_func(x, **params)
        )

        # Smart input selection for the test set
        if hasattr(self.vectorizer, "text_column"):
            test_input = pd.DataFrame({text_col: clean_test_series})
        else:
            test_input = clean_test_series

        # 2. Vectorize (transform only, no fitting)
        X_test = self.vectorizer.transform(test_input)
        y_test = test_data[label_col]

        # 3. Predict
        y_pred_test = self.best_model.predict(X_test)

        # 4. Print classification report
        print("\n[Test Set Results]")
        print(classification_report(y_test, y_pred_test, zero_division=0))

        # 5. Plot confusion matrix
        self._plot_test_confusion_matrix(y_test, y_pred_test)

        return y_test, y_pred_test

    def _plot_test_confusion_matrix(self, y_true, y_pred):
        """
        Generates a confusion matrix heatmap for the test set predictions.
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Greens',
            xticklabels=self.best_model.classes_,
            yticklabels=self.best_model.classes_
        )
        plt.title(f"Confusion Matrix (Test Set) - {self.experiment_name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()