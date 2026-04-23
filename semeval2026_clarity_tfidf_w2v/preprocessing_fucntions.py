
# Preprocessing functions
import nltk
import pandas as pd
from nltk.stem import SnowballStemmer
import re
import spacy
import emoji
import contractions
from nltk.probability import FreqDist
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from spellchecker import SpellChecker
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
stemmer = SnowballStemmer("english")



# Cleans and preprocesses text based on the specified options.



def clean_text(
    text,
    lowercase=False,
    expand_contractions=False,
    remove_urls=False,
    remove_numbers=False,
    remove_non_letters=False,
    normalize_spaces=False,
    remove_repeated_chars=False,
    tokenize=False,
    remove_short_words=False,
    min_word_length=1,        
    remove_stopwords=False,
    custom_stopwords=None,
    lemmatize=False,
    stemming=False,
    remove_duplicates=False,
):
    # --- String Operations ---
    if lowercase:
        text = text.lower()
    if expand_contractions:
        text = contractions.fix(text)
    if remove_urls:
        text = re.sub(r"http\S+|www\S+", " ", text)
    if remove_repeated_chars:
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    if remove_numbers:
        text = re.sub(r"\d+", " ", text)
    if remove_non_letters:
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
    if normalize_spaces:
        text = re.sub(r"\s+", " ", text).strip()

    needs_tokens = tokenize or remove_short_words or remove_stopwords or lemmatize or stemming or remove_duplicates
    if not needs_tokens:
        return text
    
    # Lemmatize 
    if lemmatize:
        doc = nlp(text)
        words = [token.lemma_ for token in doc if not token.is_space]
    else:
        words = text.split()
    # Stopwords
    if remove_stopwords:
        if custom_stopwords is not None:
            stop_words = set(custom_stopwords)
        else:
            stop_words = set(nlp.Defaults.stop_words)
        words = [w for w in words if w.lower() not in stop_words]    
    # Stemming 
    if stemming:
        words = [stemmer.stem(w) for w in words]

    # Short words
    if remove_short_words:
        words = [w for w in words if len(w) >= min_word_length]

    # Duplicates
    if remove_duplicates:
        words = list(dict.fromkeys(words))

    return words if tokenize else " ".join(words)





# Scans a collection of texts for common noise, 
# specific patterns, and emojis, and prints a summary report.

def detect_text_patterns(data):

    # Convert to list if the input is a Pandas Series
    if isinstance(data, pd.Series):
        texts = data.dropna().astype(str).tolist()
    else:
        texts = [str(t) for t in data if t is not None]

    # Dictionary containing the Regex patterns
    patterns = {
        "HTML Tags": r'<[^>]+>',
        "Emails": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        "URLs (http/www)": r'http\S+|www\S+',
        "Twitter Mentions (@)": r'@\w+',
        "Hashtags (#)": r'#\w+',
        "Digits": r'\d+',
        "Newlines (\\n)": r'\n'  
                    }
    print("Starting dataset inspection...\n" + "-" * 40)
    
    #Check for Regex patterns
    for name, pattern in patterns.items():
        compiled_regex = re.compile(pattern)
        matched_texts = [text for text in texts if compiled_regex.search(text)]
        match_count = len(matched_texts)
        
        if match_count > 0:
            print(f"Found: {match_count} texts with {name}.")
        else:
            print(f"Clean: No {name} found.\n")

    # Check for Emojis
    # emoji.emoji_count(text) counts how many emojis are in a text
    matched_emojis = [text for text in texts if emoji.emoji_count(text) > 0]
    emoji_count = len(matched_emojis)
    
    if emoji_count > 0:
        print(f"Found: {emoji_count} texts with Emojis.")
    else:
        print(f"Clean: No Emojis found.\n")
            
    print("-" * 40 + "\nInspection completed!")




def show_top_ngrams_by_label(
    df,
    text_column="clean_text",
    label_column="clarity_label",
    n=2,
    top_n=20
):
    for label, group in df.groupby(label_column):

        print(f"\n===== Label: {label} =====")

        ngram_list = []

        for text in group[text_column].fillna(""):
            tokens = nltk.word_tokenize(str(text))

            for i in range(len(tokens) - n + 1):
                ngram_list.append(" ".join(tokens[i:i+n]))

        fdist = FreqDist(ngram_list)
        top_items = fdist.most_common(top_n)

        for i, (ngram, freq) in enumerate(top_items, 1):
            print(f"{i:>2}. {ngram:<30} ({freq})")







def wordcloud_analysis(df, text_column="clean_text", label_column="clarity_label", category=None):

    if category is None:
        # Wordcloud for all the dataset
        text = " ".join(df[text_column].fillna(""))

        wordcloud = WordCloud(
            width=1000,
            height=500,
            background_color="white"
        ).generate(text)

        plt.figure(figsize=(12,6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("WordCloud - All Data", fontsize=18)
        plt.show()

    else:
        # Wordcloud for a specific category
        filtered_df = df[df[label_column] == category]

        text = " ".join(filtered_df[text_column].fillna(""))

        wordcloud = WordCloud(
            width=1000,
            height=500,
            background_color="white"
        ).generate(text)

        plt.figure(figsize=(12,6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"WordCloud - {category}", fontsize=18)
        plt.show()





def check_spelling_errors(text_series, examples_n=5):
    spell = SpellChecker()

    total_words = 0
    total_misspelled = 0
    examples = []

    for text in text_series.dropna():
        words = re.findall(r"\b[a-zA-Z]+\b", str(text).lower())
        total_words += len(words)

        for word in words:
            if word in spell.unknown([word]):
                total_misspelled += 1

                if len(examples) < examples_n:
                    examples.append(word)

    percentage = (total_misspelled / total_words * 100) if total_words > 0 else 0

    print(f"Total words: {total_words}")
    print(f"Misspelled words: {total_misspelled}")
    print(f"Percentage of misspelled words: {percentage:.2f}%")
    print(f"Example misspelled words: {examples}")



def find_bad_samples(text_series, threshold=30):
    spell = SpellChecker()

    for idx, text in text_series.dropna().items():
        words = re.findall(r"\b[a-zA-Z]+\b", str(text).lower())
        total_words = len(words)

        if total_words == 0:
            continue

        unknown_words = spell.unknown(words)
        misspelled_words = [w for w in words if w in unknown_words]
        misspelled_pct = (len(misspelled_words) / total_words) * 100

        if misspelled_pct > threshold:
            print(f"Sample index: {idx}")
            print(f"Misspelled percentage: {misspelled_pct:.2f}%")
            print(f"Misspelled words: {misspelled_words[:10]}")
            print(f"Text: {text}")
            print("-" * 80)



def add_word_count(df, column_name="interview_answer", new_col="answer_word_count"):
    df = df.copy()
    df[new_col] = df[column_name].astype(str).str.split().str.len()
    return df


def print_overall_stats(df, count_col):
    print("=== Overall statistics ===")
    print(df[count_col].describe())
    print("Smallest answer:", df[count_col].min())
    print("Largest answer:", df[count_col].max())


def plot_histogram(df, count_col, title, bins=30):
    plt.figure(figsize=(8, 5))
    plt.hist(df[count_col], bins=bins, edgecolor="black")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.show()


def print_label_stats(df, label_col, count_col):
    print(f"\n=== Statistics by {label_col} ===")
    print(
        df.groupby(label_col)[count_col].agg(
            ["count", "mean", "median", "min", "max", "std"]
        )
    )

def plot_clarity_boxplot(df, count_col="answer_word_count", label_col="clarity_label"):
    plt.figure(figsize=(8, 5))
    df.boxplot(column=count_col, by=label_col)
    plt.title("Word Count per Clarity Label")
    plt.suptitle("") 
    plt.xlabel("Clarity Label")
    plt.ylabel("Number of Words")
    plt.show()