# Libraries 

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict, learning_curve
from sklearn.metrics import classification_report, confusion_matrix


############################################################################################3

class TFIDFProcessor:
    def __init__(self, **tfidf_params):
        """
        Handles the conversion of text into TF-IDF features.
        Accepts any parameter of TfidfVectorizer.
        """
        self.vectorizer = TfidfVectorizer(**tfidf_params)
        self.feature_names = None
        self.is_fitted = False

    def fit_transform(self, texts):
        """Fits on the texts (e.g., Train set) and transforms them."""
        X = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.is_fitted = True
        return X

    def transform(self, texts):
        """Transforms new texts (e.g., Test set) using the already fitted vocabulary."""
        if not self.is_fitted:
            raise ValueError("Call fit_transform() first on your training data.")
        return self.vectorizer.transform(texts)

    def print_summary(self, X):
        """Prints basic statistics for the generated matrix."""
        if not self.is_fitted:
            raise ValueError("Vocabulary not built yet. Call fit_transform() first.")
        
        print("=== TF-IDF Summary ===")
        print(f"Vocabulary size: {len(self.feature_names)}")
        print(f"Feature Matrix Size (Samples x Features): {X.shape}")

class TFIDFExplorer:
    def __init__(self, X, y, feature_names, num_top_words=30, num_random_words=50, random_state=42):
        """
        Handles the analysis and exploration of the precomputed TF-IDF features.
        """
        self.X = X
        self.y = np.array(y)
        self.feature_names = np.array(feature_names)  # Needed as a numpy array for easy indexing
        self.num_top_words = num_top_words
        self.num_random_words = num_random_words
        self.random_state = random_state

    def print_random_words(self):
        rng = np.random.default_rng(self.random_state)
        n_words = min(self.num_random_words, len(self.feature_names))
        random_words = rng.choice(self.feature_names, n_words, replace=False)
        print("\n=== Random Words from Vocabulary ===")
        print(random_words)

    def print_top_words_overall(self):
        mean_tfidf = np.asarray(self.X.mean(axis=0)).flatten()
        
        df = pd.DataFrame({
            "Word": self.feature_names,
            "Mean_TFIDF": mean_tfidf
        })
        
        top_words = df.sort_values(by="Mean_TFIDF", ascending=False).head(self.num_top_words).reset_index(drop=True)
        print(f"\n=== Top {self.num_top_words} Words Overall ===")
        print(top_words)

    def print_top_words_by_label(self):
        print(f"\n=== Top {self.num_top_words} Words by Label ===")
        
        for label in sorted(np.unique(self.y)):
            # Find which rows of X belong to this label
            label_indices = np.where(self.y == label)[0]
            label_tfidf = self.X[label_indices]
            
            # Compute the mean TF-IDF only for those rows
            mean_tfidf_label = np.asarray(label_tfidf.mean(axis=0)).flatten()
            
            df = pd.DataFrame({
                "Word": self.feature_names,
                "Mean_TFIDF": mean_tfidf_label
            })
            
            top_words = df.sort_values(by="Mean_TFIDF", ascending=False).head(self.num_top_words).reset_index(drop=True)
            print(f"\n-- Label: '{label}' --")
            print(top_words)

    def report(self):
        """Runs all print functions together."""
        self.print_random_words()
        self.print_top_words_overall()
        self.print_top_words_by_label()


#####################################################################


class ModelEvaluator:
    def __init__(self, model, cv_folds=5, random_state=42):
        self.model = model
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        self.X = None
        self.y = None
        self.classes_ = None
        self.scores = None
        self.y_pred_cv = None

    def run_evaluation(self, X, y, print_report=True):
        """Cross-validation."""
        self.X = X
        self.y = y
        self.classes_ = np.unique(y)
        
        scoring_metrics = [
            "accuracy", "precision_macro", "precision_weighted",
            "recall_macro", "recall_weighted", "f1_macro", "f1_weighted"
        ]

        print(f"\n{'=' * 70}\nRunning {self.cv_folds}-fold Cross-Validation...\n{'=' * 70}")
        
        self.scores = cross_validate(self.model, self.X, self.y, cv=self.cv, scoring=scoring_metrics, n_jobs=-1)
        self.y_pred_cv = cross_val_predict(self.model, self.X, self.y, cv=self.cv, n_jobs=-1)

        self._print_summary()
        
        if print_report:
            print("\nClassification Report\n" + "-" * 70)
            print(classification_report(self.y, self.y_pred_cv, zero_division=0))

    def _print_summary(self):
        """Prints the evaluation metrics."""
        print("\nResults (Mean ± Std)")
        print("-" * 70)
        for metric in [k for k in self.scores.keys() if k.startswith("test_")]:
            m_mean = self.scores[metric].mean()
            m_std = self.scores[metric].std()
            clean_name = metric.replace('test_', '').replace('_', ' ').capitalize()
            print(f"{clean_name:20}: {m_mean:.4f} ± {m_std:.4f}")

    def plot_all(self):
        """Displays the Confusion Matrix and Learning Curve in one figure."""
        if self.y_pred_cv is None:
            print("Error: run .run_evaluation(X, y) first!")
            return

        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        
        self._plot_confusion_matrix(ax[0])
        self._plot_learning_curve(ax[1])

        plt.tight_layout()
        plt.show()

    def _plot_confusion_matrix(self, ax):
        cm = confusion_matrix(self.y, self.y_pred_cv)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.classes_, yticklabels=self.classes_, ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    def _plot_learning_curve(self, ax):
        train_sizes, train_scores, val_scores = learning_curve(
            self.model, self.X, self.y, cv=self.cv, scoring="f1_macro", n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
        val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1)

        ax.plot(train_sizes, train_mean, label="Training score", color='navy')
        ax.plot(train_sizes, val_mean, label="Validation score", color='darkorange')
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='navy')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='darkorange')

        ax.set_title("Learning Curve (F1 Macro)")
        ax.set_xlabel("Training Samples")
        ax.set_ylabel("Score")
        ax.legend()
        ax.grid(True, alpha=0.3)
#####################################################################



class ModelEvaluator:
    def __init__(self, model, cv_folds=5, random_state=42):
        self.model = model
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        self.X = None
        self.y = None
        self.classes_ = None
        self.scores = None
        self.y_pred_cv = None

    def run_evaluation(self, X, y, print_report=True):
        """ Cross-Validation"""
        self.X = X
        self.y = y
        self.classes_ = np.unique(y)
        
        scoring_metrics = [
            "accuracy", "precision_macro", "precision_weighted",
            "recall_macro", "recall_weighted", "f1_macro", "f1_weighted"
        ]

        print(f"\n{'=' * 70}\nRunning {self.cv_folds}-fold Cross-Validation...\n{'=' * 70}")
        
        self.scores = cross_validate(self.model, self.X, self.y, cv=self.cv, scoring=scoring_metrics, n_jobs=-1)
        self.y_pred_cv = cross_val_predict(self.model, self.X, self.y, cv=self.cv, n_jobs=-1)

        self._print_summary()
        
        if print_report:
            print("\nClassification Report\n" + "-" * 70)
            print(classification_report(self.y, self.y_pred_cv, zero_division=0))

    def _print_summary(self):
        """Print metrics function"""
        print("\nResults (Mean ± Std)")
        print("-" * 70)
        for metric in [k for k in self.scores.keys() if k.startswith("test_")]:
            m_mean = self.scores[metric].mean()
            m_std = self.scores[metric].std()
            clean_name = metric.replace('test_', '').replace('_', ' ').capitalize()
            print(f"{clean_name:20}: {m_mean:.4f} ± {m_std:.4f}")

    def plot_all(self):
        """ Display Confusion Matrix, Learning Curve in one figure."""
        if self.y_pred_cv is None:
            print(" Error first run .run_evaluation(X, y)!")
            return

        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        
        self._plot_confusion_matrix(ax[0])
        self._plot_learning_curve(ax[1])

        plt.tight_layout()
        plt.show()

    def _plot_confusion_matrix(self, ax):
        cm = confusion_matrix(self.y, self.y_pred_cv)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.classes_, yticklabels=self.classes_, ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    def _plot_learning_curve(self, ax):
        train_sizes, train_scores, val_scores = learning_curve(
            self.model, self.X, self.y, cv=self.cv, scoring="f1_macro", n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
        val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1)

        ax.plot(train_sizes, train_mean, label="Training score", color='navy')
        ax.plot(train_sizes, val_mean, label="Validation score", color='darkorange')
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='navy')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='darkorange')

        ax.set_title("Learning Curve (F1 Macro)")
        ax.set_xlabel("Training Samples")
        ax.set_ylabel("Score")
        ax.legend()
        ax.grid(True, alpha=0.3)




def gridsearch_logistic(
    X,
    y,
    cv_folds=5,
    max_iter=2000,
    random_state=42
):

    # Base model
    model = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state
    )

    param_grid = [
    # 1.  Liblinear:  l1/l2  'ovr' 
    {
        'C': [ 0.1, 1, 10, 30],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
        'class_weight': [None, 'balanced']
    },
    
    # 2.  SAGA:  l1/l2  'multinomial' 
    {
        'C': [ 0.1, 1, 10, 30],
        'penalty': ['l1', 'l2'],
        'solver': ['saga'],
        'class_weight': [None, 'balanced']
    },
    
    # 3.  LBFGS: only l2, 
    {
        'C': [ 0.1, 1, 10, 30],
        'penalty': ['l2'],
        'solver': ['lbfgs'],
        'class_weight': [None, 'balanced']
    }
]

    # Cross-validation strategy
    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state
    )

    # Grid Search
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1
    )

    # Fit
    grid.fit(X, y)

    # Results
    print("\nBest Parameters:")
    print(grid.best_params_)

    print("\nBest CV Score:")
    print(f"{grid.best_score_:.4f}")

    return grid.best_estimator_


