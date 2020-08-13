from gensim.scripts.glove2word2vec import glove2word2vec

if __name__ == '__main__':
    glove_input_file = 'glove.twitter.27B/glove.twitter.27B.25d.txt'
    word2vec_output_file = 'glove.twitter.27B/glove.twitter.27B.25d.word2vec.txt'
    (count, dimensions) = glove2word2vec(glove_input_file, word2vec_output_file)
    print(count, '\n', dimensions)