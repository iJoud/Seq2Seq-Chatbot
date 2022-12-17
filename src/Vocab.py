SOS_token = 0
EOS_token = 1

class Vocab:
    def __init__(self):
        self.word2index = {"": SOS_token, "": EOS_token}
        self.index2word = {SOS_token: "", EOS_token: ""}
        self.words_count = len(self.word2index)

    def add_words(self, sentence):
        for word in sentence.split(" "):
            if word not in self.word2index:
                self.word2index[word] = self.words_count
                self.index2word[self.words_count] = word
                self.words_count += 1
