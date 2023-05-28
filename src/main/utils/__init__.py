import re
import string
import collections
import random
import math
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE

data_index = 0

async def extract_words(input):
    words = list()
    for item in input:
        q_words = item.split()
        for word in q_words:
            words.append(word)
    return words

async def build_dataset(words, vocabulary_size):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count = unk_count + 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

async def generate_batch(batch_size, num_skips, skip_window, data):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1 # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [ skip_window ]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

final_words = []
# Create a custom standardization function to strip HTML break tags '<br />'.
async def train_ml_model(input_data) -> string:
    global final_words
    # Data preprocessing
    input_data = tf.strings.lower(input_data)
    input_data = tf.strings.regex_replace(input_data, '<br />', ' ')
    input_data = tf.strings.regex_replace(input_data,
                                    '[%s]' % re.escape(string.punctuation), '')
    
    vocabulary_size = 50000
    words = await extract_words(input_data.numpy())
    data, count, dictionary, reverse_dictionary = await build_dataset(words, vocabulary_size)
    del words
    for num_skips, skip_window in [(2, 1), (4, 2)]:
        data_index = 0
        batch, labels = await generate_batch(batch_size=8, num_skips=num_skips, skip_window=skip_window, data=data)

    batch_size = 128
    embedding_size = 128 
    skip_window = 1  
    num_skips = 2  
    valid_size = 16 
    valid_window = 100  
    valid_examples = np.array(random.sample(range(valid_window), valid_size))
    num_sampled = 64 

    graph = tf.Graph()

    with graph.as_default(), tf.device('/cpu:0'):
        # Input data.
        train_dataset = tf.compat.v1.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.compat.v1.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # Variables.
        embeddings = tf.Variable(
            tf.compat.v1.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        softmax_weights = tf.Variable(
            tf.compat.v1.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

        # Model.
        # Look up embeddings for inputs.
        embed = tf.nn.embedding_lookup(embeddings, train_dataset)
        # Compute the softmax loss, using a sample of the negative labels each time.
        loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=embed,
                                    labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size))

        # Optimizer.
        # Note: The optimizer will optimize the softmax_weights AND the embeddings.
        # This is because the embeddings are defined as a variable quantity and the
        # optimizer's `minimize` method will by default modify all variable quantities
        # that contribute to the tensor it is passed.
        # See docs on `tf.train.Optimizer.minimize()` for more details.
        optimizer = tf.compat.v1.train.AdagradOptimizer(1.0).minimize(loss)

        # Compute the similarity between minibatch examples and all embeddings.
        # We use the cosine distance:
        norm = tf.compat.v1.sqrt(tf.compat.v1.reduce_sum(tf.compat.v1.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))
        print("Shapes of tensors similarity, embeddings, norm, normalized_embeddings, valid_embeddings")
        print(similarity)
        print(embeddings)
        print(norm)
        print(normalized_embeddings)
        print(valid_embeddings)
    num_steps = 100001

    with tf.compat.v1.Session(graph=graph) as session:
        tf.compat.v1.global_variables_initializer().run()
        print('Initialized')
        average_loss = 0
        for step in range(num_steps):
            batch_data, batch_labels = await generate_batch(
            batch_size, num_skips, skip_window, data)
            feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
            _, l = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += l
            if step % 2000 == 0:
                if step > 0:
                    average_loss = average_loss / 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step %d: %f' % (step, average_loss))
            average_loss = 0
            # note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 10000 == 0:
                sim = similarity.eval()
                for i in range(len(valid_examples)):
                    if valid_examples[i] not in reverse_dictionary:
                        continue
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8
                    nearest = (-sim[i, :]).argsort()[1:top_k+1]
                    log = 'Nearest to %s:' % valid_word
                    print(nearest)
                    print(len(reverse_dictionary))
                    for k in range(len(nearest)):
                        if nearest[k] not in reverse_dictionary:
                            continue
                        a = nearest[k]
                        print(a)
                        close_word = reverse_dictionary[nearest[k]]
                        print(close_word)
                        log = '%s %s,' % (log, close_word)
                        print(log)
        final_embeddings = normalized_embeddings.eval()
    num_points = 400

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points+1, :])
    return two_d_embeddings