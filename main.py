import numpy as np
import matplotlib.pyplot as plt
import nltk
import warnings

# Suppress NLTK downloads and runtime warnings
warnings.filterwarnings("ignore")
nltk.download('treebank', quiet=True)

# Load tagged sentences from the treebank corpus
tagged_sentences = nltk.corpus.treebank.tagged_sents()
ct = nltk.tag.CRFTagger()

# Train the CRF tagger using the tagged sentences
ct.train(tagged_sentences, 'model.crf.tagger')

text_to_tag = ("The non-zero-sum game element emerges as the communication progresses. Instead of a traditional "
               "winner-takes-all scenario, the goal becomes mutual understanding and cooperation. Learning the "
               "Heptapod language alters Dr. Banks' perception of time, enabling her to foresee future events. This "
               "knowledge doesn't result in the dominance of one side over the other; instead, it becomes a tool for "
               "shared comprehension and collaboration.")

tokenized_text = nltk.word_tokenize(text_to_tag)

# Load the trained model (assuming it's already trained and saved)
ct.set_model_file('model.crf.tagger')

# Tag the text using the trained CRF tagger
tagged_text = ct.tag(tokenized_text)

# Extract POS tags from the tagged text
pos_tags = [tag for word, tag in tagged_text]

# Generate unique tags and their counts
unique_tags, tag_counts = np.unique(pos_tags, return_counts=True)

# Mapping POS tags to descriptions
tag_descriptions = {
    'DT': 'Determiner',
    'IN': 'Preposition or Subordinating Conjunction',
    'JJ': 'Adjective',
    'NN': 'Noun (Singular or Mass)',
    'NNS': 'Noun (Plural)',
    'RB': 'Adverb',
    'VB': 'Verb (Base Form)',
    'VBG': 'Verb (Gerund or Present Participle)',
    'VBZ': 'Verb (Third Person Singular Present)',
    'CC': 'Coordinating Conjunction',
    'CD': 'Cardinal Number',
    'NNP': 'Proper Noun (Singular)',
    'POS': 'Possessive Ending',
    'PRP': 'Personal Pronoun',
    ',': 'Comma',
    '.': 'Period',
    ':': 'Colon',
    ';': 'Semicolon',
    'TO': 'to (Preposition or Infinitive Marker)'
}

# Mapping POS tags to their descriptions
pos_with_desc = [tag_descriptions.get(tag, 'Unknown') for tag in unique_tags]

# Visualizing the distribution of POS tags
plt.figure(figsize=(10, 6))
plt.barh(pos_with_desc, tag_counts, color='purple')
plt.xlabel('Counts')
plt.ylabel('POS Descriptions')
plt.title('Distribution of POS Tags with Descriptions')
plt.tight_layout()
plt.show()

# Print explanations for each POS tag in the tagged text
for word, tag in tagged_text:
    print(f"{word}: {tag} - {tag_descriptions.get(tag, 'Unknown')}")

