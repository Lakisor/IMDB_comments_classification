import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
from wordcloud import WordCloud

# Set style for plots
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_data(file_path):
    return pd.read_csv(file_path)

def basic_info(df):
    print("\n=== Dataset Information ===")
    print(f"Total samples: {len(df)}")
    print("\n=== First 5 rows ===")
    print(df.head())
    print("\n=== Data Types ===")
    print(df.dtypes)
    print("\n=== Missing Values ===")
    print(df.isnull().sum())

def analyze_sentiment_distribution(df, sentiment_col='sentiment'):
    print("\n=== Sentiment Distribution ===")
    print(df[sentiment_col].value_counts())
    
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x=sentiment_col, data=df)
    plt.title('Distribution of Sentiments')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    
    # Add count labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', 
                   xytext=(0, 10), 
                   textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig('data/analysis/sentiment_distribution.png')
    plt.close()

def analyze_text_length(df, text_col='review'):
    print("\n=== Text Length Analysis ===")
    df['text_length'] = df[text_col].apply(len)
    
    print("\nText Length Statistics (characters):")
    print(df['text_length'].describe())
    
    # Plot text length distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='text_length', bins=50, kde=True)
    plt.title('Distribution of Text Lengths')
    plt.xlabel('Number of Characters')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('data/analysis/text_length_distribution.png')
    plt.close()
    
    # Plot text length by sentiment
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='sentiment', y='text_length')
    plt.title('Text Length by Sentiment')
    plt.tight_layout()
    plt.savefig('data/analysis/text_length_by_sentiment.png')
    plt.close()

def analyze_word_frequencies(df, text_col='review', sentiment_col='sentiment', top_n=20):
    print("\n=== Word Frequency Analysis ===")
    
    os.makedirs('data/analysis/wordclouds', exist_ok=True)
    
    def get_words(text):
        words = text.lower().split()
        words = [word.strip('.,!?"\'()[]{}:;') for word in words]
        return [word for word in words if word.isalpha() and len(word) > 2]
    
    for sentiment in df[sentiment_col].unique():
        print(f"\nAnalyzing {sentiment} reviews...")
        sentiment_df = df[df[sentiment_col] == sentiment]
        
        all_words = []
        for text in sentiment_df[text_col]:
            all_words.extend(get_words(text))
        
        word_freq = Counter(all_words)
        
        print(f"\nTop {top_n} words in {sentiment} reviews:")
        for word, count in word_freq.most_common(top_n):
            print(f"{word}: {count}")
        
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_words))
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Word Cloud - {sentiment.capitalize()} Reviews')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'data/analysis/wordclouds/wordcloud_{sentiment}.png')
        plt.close()

def save_analysis_report(df, output_dir='data/analysis'):
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== Dataset Analysis Report ===\n\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Number of features: {len(df.columns)}\n\n")
        
        f.write("=== Sentiment Distribution ===\n")
        f.write(df['sentiment'].value_counts().to_string() + "\n\n")
        
        if 'text_length' in df.columns:
            f.write("=== Text Length Statistics (characters) ===\n")
            f.write(df['text_length'].describe().to_string() + "\n")
        
        f.write("\n=== Missing Values ===\n")
        f.write(df.isnull().sum().to_string() + "\n")
    
    print(f"\nAnalysis report saved to: {report_path}")

if __name__ == "__main__":
    os.makedirs('data/analysis', exist_ok=True)
    
    input_file = "IMDB Dataset.csv"
    print(f"Loading dataset from: {input_file}")
    
    try:
        df = load_data(input_file)
        
        basic_info(df)
        
        analyze_sentiment_distribution(df)
        
        analyze_text_length(df)
        
        analyze_word_frequencies(df)
        
        save_analysis_report(df)
        
        print("\nAnalysis completed successfully! Check the 'data/analysis' directory for results.")
        
    except FileNotFoundError:
        print(f"Error: Input data file not found at {input_file}")
        print("Please ensure the data file exists in the current directory.")
    except Exception as e:
        print(f"An error occurred during analysis: {str(e)}")
