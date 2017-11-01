import argparse
import numpy as np
import random

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

"""
Pretrained word vectors: https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
"""

# Constants from C++ code
EMBEDDING_DIM = 64
LAYERS = 1
INPUT_DIM = 64
XCRIBE_DIM = 64 # 32
SEG_DIM = 16
H1DIM = 32
H2DIM = 32
TAG_DIM = 32
DURATION_DIM = 4

# lstm builder: LAYERS, XCRIBE_DIM, SEG_DIM, m?
# (layers, input_dim, hidden_dim, model)

DATA_MAX_SEG_LEN = 15

# not used
MAX_SENTENCE_LEN = 200

use_pretrained_embeding = False
use_dropout = False
dropout_rate = 0.0
ner_tagging = False
pretrained_embeding = ""

LABELS = ['DET', 'AUX', 'ADJ', 'ADP', 'VERB', 'NOUN', 'SYM', 'PROPN', 'PART', 'X', 'CCONJ', 'PRON', 'ADV', 'PUNCT', 'NUM']

def logsumexp(inputs, dim=None, keepdim=False):
        return (inputs - F.log_softmax(inputs)).mean(dim, keepdim=keepdim)

# SegRNN module
class SegRNN(nn.Module):
    def __init__(self):
        super(SegRNN, self).__init__()
        self.forward_initial = (nn.Parameter(torch.randn(1, 1, SEG_DIM)), nn.Parameter(torch.randn(1, 1, SEG_DIM)))
        self.backward_initial = (nn.Parameter(torch.randn(1, 1, SEG_DIM)), nn.Parameter(torch.randn(1, 1, SEG_DIM)))
        self.Y_encoding = [nn.Parameter(torch.randn(1, 1, TAG_DIM)) for i in range(len(LABELS))]
        self.Z_encoding = [nn.Parameter(torch.randn(1, 1, DURATION_DIM)) for i in range(1, DATA_MAX_SEG_LEN + 1)]

        self.register_parameter("forward_initial_0", self.forward_initial[0])
        self.register_parameter("forward_initial_1", self.forward_initial[1])
        self.register_parameter("backward_initial_0", self.backward_initial[0])
        self.register_parameter("backward_initial_1", self.backward_initial[1])
        for idx, encoding in enumerate(self.Y_encoding):
            self.register_parameter("Y_encoding_" + str(idx), encoding)
        for idx, encoding in enumerate(self.Z_encoding):
            self.register_parameter("Z_encoding_" + str(idx), encoding)

        self.forward_lstm = nn.LSTM(XCRIBE_DIM, SEG_DIM)
        self.backward_lstm = nn.LSTM(XCRIBE_DIM, SEG_DIM)
        self.V = nn.Linear(SEG_DIM + SEG_DIM + TAG_DIM + DURATION_DIM, SEG_DIM)
        self.W = nn.Linear(SEG_DIM, 1)
        self.Phi = nn.Tanh()

    def calc_loss(self, data, label):
        N, K = data.shape

        forward_precalc = [[None for _ in range(N)] for _ in range(N)]
        # forward_precalc[i, j, :] => [i, j]
        for i in range(N):
            hidden = self.forward_initial
            for j in range(i, min(N, i + DATA_MAX_SEG_LEN)):
                next_input = autograd.Variable(torch.from_numpy(data[j, :]).float())
                out, hidden = self.forward_lstm(next_input.view(1, 1, K), hidden)
                forward_precalc[i][j] = out

        backward_precalc = [[None for _ in range(N)] for _ in range(N)]
        # backward_precalc[i, j, :] => [i, j]
        for i in range(N):
            hidden = self.backward_initial
            for j in range(i, max(-1, i - DATA_MAX_SEG_LEN), -1):
                next_input = autograd.Variable(torch.from_numpy(data[j, :]).float())
                out, hidden = self.backward_lstm(next_input.view(1, 1, K), hidden)
                backward_precalc[j][i] = out

        log_alphas = [autograd.Variable(torch.from_numpy(np.array([0.0])).float())]
        for i in range(1, N + 1):
            t_sum = []
            for j in range(max(0, i - DATA_MAX_SEG_LEN), i):
                precalc_expand = torch.cat([forward_precalc[j][i - 1], backward_precalc[j][i - 1]], 2).repeat(len(LABELS), 1, 1)
                y_encoding_expand = torch.cat([self.Y_encoding[y] for y in range(len(LABELS))], 0)
                z_encoding_expand = torch.cat([self.Z_encoding[i - j - 1] for y in range(len(LABELS))])
                seg_encoding = torch.cat([precalc_expand, y_encoding_expand, z_encoding_expand], 2)
                t = logsumexp(self.W(self.Phi(self.V(seg_encoding))))
                t_sum.append(log_alphas[j] + t)
            log_alphas.append(logsumexp(torch.cat(t_sum)))

        indiv = autograd.Variable(torch.zeros(1))
        chars = 0
        for tag, length in label:
            if length >= DATA_MAX_SEG_LEN:
                continue
            seg_encoding = torch.cat([forward_precalc[chars][chars + length - 1], backward_precalc[chars][chars + length - 1], self.Y_encoding[LABELS.index(tag)], self.Z_encoding[length - 1]], 2)
            indiv += self.W(self.Phi(self.V(seg_encoding)))
        loss = log_alphas[N] - indiv
        #print(log_alphas[N], indiv)
        return loss

    def _precalc(self, data):
        N, K = data.shape
        forward_precalc = [[None for _ in range(N)] for _ in range(N)]
        # forward_precalc[i, j, :] => [i, j]
        for i in range(N):
            hidden = self.forward_initial
            for j in range(i, min(N, i + DATA_MAX_SEG_LEN)):
                next_input = autograd.Variable(torch.from_numpy(data[j, :]).float())
                out, hidden = self.forward_lstm(next_input.view(1, 1, K), hidden)
                forward_precalc[i][j] = out

        backward_precalc = [[None for _ in range(N)] for _ in range(N)]
        # backward_precalc[i, j, :] => [i, j]
        for i in range(N):
            hidden = self.backward_initial
            for j in range(i, max(-1, i - DATA_MAX_SEG_LEN), -1):
                next_input = autograd.Variable(torch.from_numpy(data[j, :]).float())
                out, hidden = self.backward_lstm(next_input.view(1, 1, K), hidden)
                backward_precalc[j][i] = out
        return forward_precalc, backward_precalc

    def infer(self, data):
        N, K = data.shape
        forward_precalc, backward_precalc = self._precalc(data)
        
        log_alphas = [(-1, -1, 0.0)]
        for i in range(1, N + 1):
            t_sum = []
            max_len = -1
            max_t = float("-inf")
            max_label = -1
            for j in range(max(0, i - DATA_MAX_SEG_LEN), i):
                precalc_expand = torch.cat([forward_precalc[j][i - 1], backward_precalc[j][i - 1]], 2).repeat(len(LABELS), 1, 1)
                y_encoding_expand = torch.cat([self.Y_encoding[y] for y in range(len(LABELS))], 0)
                z_encoding_expand = torch.cat([self.Z_encoding[i - j - 1] for y in range(len(LABELS))])
                seg_encoding = torch.cat([precalc_expand, y_encoding_expand, z_encoding_expand], 2)
                t = self.W(self.Phi(self.V(seg_encoding))) + log_alphas[j][2]
                for y in range(len(LABELS)):
                    if t.data[y, 0, 0] > max_t:
                        max_t = t.data[y, 0, 0]
                        max_label = y
                        max_len = i - j
            log_alphas.append((max_label, max_len, max_t))
        
        cur_pos = N
        ret = []
        while cur_pos != 0:
            ret.append((LABELS[log_alphas[cur_pos][0]], log_alphas[cur_pos][1]))
            cur_pos -= log_alphas[cur_pos][1]
        return list(reversed(ret))

def parse_embedding(embed_filename):
    embed_file = open(embed_filename)
    embedding = dict()
    for line in embed_file:
        values = line.split()
        embedding[values[0]] = np.array(values[1:]).astype(np.float)
    return embedding

def parse_file(train_filename, embedding):
    train_file = open(train_filename)
    sentences = []
    labels = []
    label = []
    POS_labels = set()
    for line in train_file:
        if line.startswith("# text = "):
            sentence = line[9:].strip()
            N = len(sentence)
            sentence_vec = np.zeros((N, EMBEDDING_DIM))
            for i in range(N):
                c = sentence[i]
                if c in embedding:
                    sentence_vec[i, :] = embedding[c]
                elif c in "0123456789":
                    sentence_vec[i, :] = embedding["<NUM>"]
                else:
                    sentence_vec[i, :] = embedding["<unk>"]
            sentences.append(sentence_vec)
        elif not line.startswith("#"):
            parts = line.split()
            if len(parts) < 4:
                labels.append(label)
                label = []
            else:
                label.append((parts[3], len(parts[1])))

    return sentences, labels

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segmental RNN.')
    parser.add_argument('--train', help='Training file')
    parser.add_argument('--embed', help='Character embedding file')
    args = parser.parse_args()

    embedding = parse_embedding(args.embed)
    print("Done parsing embedding")
    data, labels = parse_file(args.train, embedding)
    pairs = list(zip(data, labels))
    print("Done parsing training data")

    seg_rnn = SegRNN()
    optimizer = torch.optim.Adam(seg_rnn.parameters(), lr=0.01)
    random.seed(1337)
    count = 0.0
    sum_loss = 0.0
    for batch_num in range(1000):
        random.shuffle(pairs)
        for i in range(len(data)):
            optimizer.zero_grad()

            datum, label = pairs[i]
            loss = seg_rnn.calc_loss(datum, label)
            sum_loss += loss.data[0]
            count += 1.0
            loss.backward()

            optimizer.step()
            if i % 10 == 0:
                print("Batch ", batch_num, " datapoint ", i, " avg loss ", sum_loss / count)
                print(seg_rnn.infer(datum))
                print(label)
                print(seg_rnn.Z_encoding[0])
                #for param in seg_rnn.parameters():
                #    print(param)
