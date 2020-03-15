import wordcloud
import matplotlib.pyplot as plt


def topic_cloud(topic_dict, max_words=20, filename=None):
    """
    :param topic_dict: The dictionary of words with their probabilities that represent the topic
                       i.e. {word1: probability1, word2: probability2,...}
    :param filename: Where to save the image
    :param max_words: up to how many words to include in the cloud representation
    :return:
    """
    weights = {word: score for word, score in topic_dict}
    wc = wordcloud.WordCloud(
        background_color="white",
        max_words=max_words
    )
    cloud = wc.generate_from_frequencies(frequencies=weights)
    if filename:
        cloud.to_file(filename)
    else:
        plt.imshow(cloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
