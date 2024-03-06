import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from heapq import nlargest

# Sample news data
news_data = """
Scientists have discovered a new species of dinosaur in Argentina, which they believe to be the largest land animal to have ever walked the Earth.

Named 'Patagotitan mayorum', the dinosaur weighed an estimated 69 tonnes and measured more than 37 meters in length. It lived about 100 million years ago during the Late Cretaceous period.

Researchers unearthed the fossils of this massive creature in 2014 in the Patagonian desert of Argentina. The discovery sheds light on the diversity and size of dinosaurs that roamed the Earth during the prehistoric era.

In another breakthrough, a team of engineers has developed a revolutionary new battery technology that promises longer-lasting and faster-charging batteries for electronic devices.

The new battery uses a combination of graphene and silicon, which enables it to store more energy and deliver power more efficiently than conventional lithium-ion batteries. This innovation could significantly improve the performance of smartphones, laptops, and electric vehicles.

Meanwhile, artificial intelligence continues to make strides in various fields, including healthcare and finance. Researchers have developed AI algorithms that can accurately diagnose medical conditions and predict market trends with unprecedented accuracy.

These advancements in AI technology are expected to revolutionize industries and improve decision-making processes in the near future. However, concerns remain about the ethical implications and potential job displacement associated with the widespread adoption of AI.

Overall, recent developments in science and technology have the potential to reshape our world and drive innovation in the years to come.
"""

# Tokenize the text into sentences
sentences = sent_tokenize(news_data)

# Tokenize the text into words
words = word_tokenize(news_data.lower())

# Remove stopwords
stop_words = set(stopwords.words("english"))
filtered_words = [word for word in words if word not in stop_words]

# Calculate word frequency
word_freq = FreqDist(filtered_words)

# Assign score to each sentence based on word frequency
sentence_scores = {}
for sentence in sentences:
    for word in word_tokenize(sentence.lower()):
        if word in word_freq.keys():
            if len(sentence.split(" ")) < 30:
                if sentence not in sentence_scores.keys():
                    sentence_scores[sentence] = word_freq[word]
                else:
                    sentence_scores[sentence] += word_freq[word]

# Get the top N sentences with highest scores
summary_sentences = nlargest(3, sentence_scores, key=sentence_scores.get)

# Join the summary sentences into a single string
summary = ' '.join(summary_sentences)

print(summary)
