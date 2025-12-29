import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

def load_text(filepath):
    """
    Load text from a file with encoding handling
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
    except UnicodeDecodeError:
        with open(filepath, 'r', encoding='ascii', errors='ignore') as file:
            text = file.read()

    return text

def basic_statistics(text):
    characters = len(text)

    words = re.findall(r'\b\w+\b', text)
    word_count = len(words)

    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if s.strip()]
    sentence_count = len(sentences)

    avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

    return {
        "characters": characters,
        "words": word_count,
        "sentences": sentence_count,
        "avg_word_length": round(avg_word_length, 2),
        "avg_sentence_length": round(avg_sentence_length, 2)
    }

def create_metadata(filename, text):
    stats = basic_statistics(text)

    data = {
        "document_id": [1],
        "filename": [filename],
        "character_count": [stats["characters"]],
        "word_count": [stats["words"]],
        "sentence_count": [stats["sentences"]]
    }

    df = pd.DataFrame(data)
    return df

def save_csv(df, output_path):
    df.to_csv(output_path, index=False)

def extract_emails(text):
    """
    Extract email addresses using regex
    """
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    emails = re.findall(pattern, text)

    return {
        "total_emails": len(emails),
        "unique_emails": list(set(emails))
    }

def extract_phone_numbers(text):
    """
    Extract phone numbers in multiple formats and normalize them
    """
    pattern = r'(\+?\d{1,2}[-\s]?)?\(?\d{3}\)?[-\s]\d{3}[-\s]\d{4}'
    raw_numbers = re.findall(pattern, text)

    normalized = []
    for num in raw_numbers:
        digits = re.sub(r'\D', '', num)
        if len(digits) >= 10:
            normalized.append(digits[-10:])

    return list(set(normalized))

def extract_urls(text):
    """
    Extract URLs and categorize by domain
    """
    pattern = r'https?://[^\s]+'
    urls = re.findall(pattern, text)

    domains = {}
    for url in urls:
        domain = re.findall(r'https?://([^/]+)/?', url)
        if domain:
            domains.setdefault(domain[0], 0)
            domains[domain[0]] += 1

    return {
        "total_urls": len(urls),
        "domains": domains
    }

def extract_dates(text):
    """
    Extract dates in multiple formats
    """
    patterns = [
        r'\b\d{2}/\d{2}/\d{4}\b',            # DD/MM/YYYY
        r'\b\d{2}-\d{2}-\d{4}\b',            # MM-DD-YYYY
        r'\b[A-Za-z]+ \d{1,2}, \d{4}\b'      # Month DD, YYYY
    ]

    dates = []
    for pattern in patterns:
        dates.extend(re.findall(pattern, text))

    return list(set(dates))

def extract_currency(text):
    """
    Extract currency amounts
    """
    pattern = r'[\$€£]\s?\d+'
    currencies = re.findall(pattern, text)

    return list(set(currencies))

def save_patterns(patterns, output_path):
    with open(output_path, 'w') as file:
        json.dump(patterns, file, indent=4)

def calculate_word_frequency(text):
    """
    Calculate word frequency dictionary
    """
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

    freq_dict = {}
    for word in words:
        freq_dict[word] = freq_dict.get(word, 0) + 1

    return freq_dict

def word_frequency_statistics(freq_dict):
    """
    Compute statistics using NumPy
    """
    frequencies = np.array(list(freq_dict.values()))

    stats = {
        "mean": np.mean(frequencies),
        "median": np.median(frequencies),
        "std_dev": np.std(frequencies),
        "25_percentile": np.percentile(frequencies, 25),
        "50_percentile": np.percentile(frequencies, 50),
        "75_percentile": np.percentile(frequencies, 75)
    }

    return stats

def character_analysis(text):
    """
    Analyze character frequency and vowel/consonant ratio
    """
    text = text.lower()
    characters = re.findall(r'[a-z]', text)

    char_freq = {}
    for char in characters:
        char_freq[char] = char_freq.get(char, 0) + 1

    vowels = set('aeiou')
    vowel_count = sum(char_freq.get(v, 0) for v in vowels)
    consonant_count = sum(char_freq.values()) - vowel_count

    ratio = round(vowel_count / consonant_count, 3) if consonant_count > 0 else 0

    return {
        "char_frequency": char_freq,
        "most_common": max(char_freq, key=char_freq.get),
        "least_common": min(char_freq, key=char_freq.get),
        "vowel_consonant_ratio": ratio
    }

def sentence_length_analysis(text):
    """
    Analyze sentence length distribution
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    sentence_lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]

    return {
        "sentence_lengths": sentence_lengths,
        "longest_sentence": max(sentence_lengths),
        "shortest_sentence": min(sentence_lengths)
    }

def plot_top_words(freq_dict, top_n=20):
    words = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    labels, values = zip(*words)

    plt.figure(figsize=(10, 6))
    plt.barh(labels[::-1], values[::-1])
    plt.xlabel("Frequency")
    plt.title("Top 20 Most Frequent Words")
    plt.tight_layout()
    plt.savefig("plots/top_20_words.png")
    plt.close()

def plot_word_length_distribution(freq_dict):
    word_lengths = []
    for word, freq in freq_dict.items():
        word_lengths.extend([len(word)] * freq)

    plt.figure(figsize=(8, 5))
    plt.hist(word_lengths, bins=20)
    plt.xlabel("Word Length")
    plt.ylabel("Frequency")
    plt.title("Word Length Distribution")
    plt.tight_layout()
    plt.savefig("plots/word_length_histogram.png")
    plt.close()

def plot_character_frequency(char_freq):
    top_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)[:15]
    labels, values = zip(*top_chars)

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.xlabel("Characters")
    plt.ylabel("Frequency")
    plt.title("Top 15 Character Frequencies")
    plt.tight_layout()
    plt.savefig("plots/character_frequency.png")
    plt.close()

def plot_sentence_length_box(sentence_lengths):
    plt.figure(figsize=(6, 4))
    plt.boxplot(sentence_lengths, vert=False)
    plt.xlabel("Words per Sentence")
    plt.title("Sentence Length Distribution")
    plt.tight_layout()
    plt.savefig("plots/sentence_length_boxplot.png")
    plt.close()

def plot_dashboard(freq_dict, char_freq, sentence_lengths):
    top_words = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    words, counts = zip(*top_words)

    plt.figure(figsize=(12, 10))

    # Plot 1: Top words
    plt.subplot(2, 2, 1)
    plt.barh(words[::-1], counts[::-1])
    plt.title("Top 10 Words")

    # Plot 2: Word length
    lengths = [len(w) for w in freq_dict.keys()]
    plt.subplot(2, 2, 2)
    plt.hist(lengths, bins=15)
    plt.title("Word Lengths")

    # Plot 3: Character frequency
    chars, char_counts = zip(*sorted(char_freq.items(), key=lambda x: x[1], reverse=True)[:10])
    plt.subplot(2, 2, 3)
    plt.bar(chars, char_counts)
    plt.title("Top Characters")

    # Plot 4: Sentence length
    plt.subplot(2, 2, 4)
    plt.boxplot(sentence_lengths, vert=False)
    plt.title("Sentence Lengths")

    plt.suptitle("Text Statistics Dashboard")
    plt.tight_layout()
    plt.savefig("plots/dashboard.png")
    plt.close()

def create_word_dataframe(freq_dict):
    data = {
        "word": [],
        "frequency": [],
        "length": [],
        "first_char": [],
        "vowel_count": []
    }

    vowels = set("aeiou")

    for word, freq in freq_dict.items():
        data["word"].append(word)
        data["frequency"].append(freq)
        data["length"].append(len(word))
        data["first_char"].append(word[0])
        data["vowel_count"].append(sum(1 for c in word if c in vowels))

    df = pd.DataFrame(data)
    df.set_index("word", inplace=True)

    return df

def groupby_analysis(df):
    by_length = df.groupby("length")["frequency"].mean()
    by_first_char = df.groupby("first_char")["frequency"].sum()

    return by_length, by_first_char

def sorting_and_filtering(df):
    top_50_words = df.sort_values(by="frequency", ascending=False).head(50)
    long_words = df[df["length"] > 10]

    return top_50_words, long_words

def create_pivot_table(df):
    pivot = pd.pivot_table(
        df,
        values="frequency",
        index="first_char",
        columns="length",
        aggfunc="mean",
        fill_value=0
    )
    return pivot

if __name__ == "__main__":

    filepath = "data/alice.txt"

    # ---------- PART 1 ----------
    text = load_text(filepath)

    stats = basic_statistics(text)
    print("Basic Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")

    metadata_df = create_metadata("alice.txt", text)
    save_csv(metadata_df, "outputs/document_stats.csv")

    # ---------- PART 2 ----------
    patterns = {
        "emails": extract_emails(text),
        "phone_numbers": extract_phone_numbers(text),
        "urls": extract_urls(text),
        "dates": extract_dates(text),
        "currency": extract_currency(text)
    }
    save_patterns(patterns, "outputs/patterns.json")

    # ---------- PART 3 ----------
    word_freq = calculate_word_frequency(text)
    freq_stats = word_frequency_statistics(word_freq)

    char_stats = character_analysis(text)
    sentence_stats = sentence_length_analysis(text)

    # ---------- PART 4 ----------
    plot_top_words(word_freq)
    plot_word_length_distribution(word_freq)
    plot_character_frequency(char_stats["char_frequency"])
    plot_sentence_length_box(sentence_stats["sentence_lengths"])
    plot_dashboard(word_freq, char_stats["char_frequency"], sentence_stats["sentence_lengths"])

    # ---------- PART 5 ----------
    word_df = create_word_dataframe(word_freq)
    by_length, by_first_char = groupby_analysis(word_df)
    top_50, long_words = sorting_and_filtering(word_df)
    pivot_table = create_pivot_table(word_df)

    print("\nProgram executed successfully ✅")

    print("\nTop 10 Words:")
    print(top_50.head(10))

    print("\nWords longer than 10 characters:")
    print(long_words.head())

print("\nPivot table created successfully")

