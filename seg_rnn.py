import argparse
import numpy as np
import random
import time

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

# Constants from C++ code
EMBEDDING_DIM = 64
LAYERS = 1
INPUT_DIM = 64
XCRIBE_DIM = 96
SEG_DIM = 16
H1DIM = 32
H2DIM = 32
TAG_DIM = 32
DURATION_DIM = 4

# lstm builder: LAYERS, XCRIBE_DIM, SEG_DIM, m?
# (layers, input_dim, hidden_dim, model)

DATA_MAX_SEG_LEN = 15

MAX_SENTENCE_LEN = 32
MINIBATCH_SIZE = 32
BATCH_SIZE = 256

use_max_sentence_len_training = True

LABELS = ['DET', 'AUX', 'ADJ', 'ADP', 'VERB', 'NOUN', 'SYM', 'PROPN', 'PART', 'X', 'CCONJ', 'PRON', 'ADV', 'PUNCT', 'NUM']

def logsumexp(inputs, dim=None, keepdim=False):
        return (inputs - F.log_softmax(inputs)).mean(dim, keepdim=keepdim)

# SegRNN module
class SegRNN(nn.Module):
    def __init__(self):
        super(SegRNN, self).__init__()
        self.forward_context_initial = (nn.Parameter(torch.randn(2, 1, XCRIBE_DIM)), nn.Parameter(torch.randn(2, 1, XCRIBE_DIM)))
        self.backward_context_initial = (nn.Parameter(torch.randn(2, 1, XCRIBE_DIM)), nn.Parameter(torch.randn(2, 1, XCRIBE_DIM)))
        self.forward_context_lstm = nn.LSTM(INPUT_DIM, XCRIBE_DIM, 2, dropout=0.4)
        self.backward_context_lstm = nn.LSTM(INPUT_DIM, XCRIBE_DIM, 2, dropout=0.4)
        self.register_parameter("forward_context_initial_0", self.forward_context_initial[0])
        self.register_parameter("forward_context_initial_1", self.forward_context_initial[1])
        self.register_parameter("backward_context_initial_0", self.backward_context_initial[0])
        self.register_parameter("backward_context_initial_1", self.backward_context_initial[1])

        self.forward_initial = (nn.Parameter(torch.randn(2, 1, SEG_DIM)), nn.Parameter(torch.randn(2, 1, SEG_DIM)))
        self.backward_initial = (nn.Parameter(torch.randn(2, 1, SEG_DIM)), nn.Parameter(torch.randn(2, 1, SEG_DIM)))
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

        self.forward_lstm = nn.LSTM(2 * XCRIBE_DIM, SEG_DIM, 2, dropout=0.4)
        self.backward_lstm = nn.LSTM(2 * XCRIBE_DIM, SEG_DIM, 2, dropout=0.4)
        self.V = nn.Linear(SEG_DIM + SEG_DIM + TAG_DIM + DURATION_DIM, SEG_DIM)
        self.W = nn.Linear(SEG_DIM, 1)
        self.Phi = nn.Tanh()

    def calc_loss(self, batch_data, batch_label):
        N, B, K = batch_data.shape
        print(B, len(batch_label))
        forward_precalc, backward_precalc = self._precalc(batch_data)

        log_alphas = [autograd.Variable(torch.zeros((1, B, 1)))]
        for i in range(1, N + 1):
            t_sum = []
            for j in range(max(0, i - DATA_MAX_SEG_LEN), i):
                precalc_expand = torch.cat([forward_precalc[j][i - 1], backward_precalc[j][i - 1]], 2).repeat(len(LABELS), 1, 1)
                y_encoding_expand = torch.cat([self.Y_encoding[y] for y in range(len(LABELS))], 0).repeat(1, B, 1)
                z_encoding_expand = torch.cat([self.Z_encoding[i - j - 1] for y in range(len(LABELS))]).repeat(1, B, 1)
                # LABELS, MINIBATCH, FEATURES
                seg_encoding = torch.cat([precalc_expand, y_encoding_expand, z_encoding_expand], 2)
                # Linear thru features: LABELS, MINIBATCH, 1
                t = self.W(self.Phi(self.V(seg_encoding)))
                # summed across labels: 1, MINIBATCH, 1
                summed_t = logsumexp(t, 0, True)
                t_sum.append(log_alphas[j] + summed_t)
            # cat across seglenths: SEG_LENGTH, MINIBATCH, 1
            all_t_sums = torch.cat(t_sum, 0)
            # sum across lengths: 1, MINIBATCH, 1
            new_log_alpha = logsumexp(all_t_sums, 0, True)
            log_alphas.append(new_log_alpha)

        loss = torch.sum(log_alphas[N])

        for batch_idx in range(B):
            indiv = autograd.Variable(torch.zeros(1))
            chars = 0
            label = batch_label[batch_idx]
            for tag, length in label:
                if length >= DATA_MAX_SEG_LEN:
                    continue
                forward_val = forward_precalc[chars][chars + length - 1][:, batch_idx, np.newaxis, :]
                backward_val = backward_precalc[chars][chars + length - 1][:, batch_idx, np.newaxis, :]
                y_val = self.Y_encoding[LABELS.index(tag)]
                z_val = self.Z_encoding[length - 1]
                seg_encoding = torch.cat([forward_val, backward_val, y_val, z_val], 2)
                indiv += self.W(self.Phi(self.V(seg_encoding)))
                chars += length
            loss -= indiv
        return loss

    def _precalc(self, data):
        N, B, K = data.shape

        forward_xcribe_data = []
        hidden = (
            torch.cat([self.forward_context_initial[0] for b in range(B)], 1),
            torch.cat([self.forward_context_initial[1] for b in range(B)], 1)
        )
        for i in range(N):
            next_input = autograd.Variable(torch.from_numpy(data[i, :]).float())
            out, hidden = self.forward_context_lstm(next_input.view(1, B, K), hidden)
            forward_xcribe_data.append(out)
        backward_xcribe_data = []
        hidden = (
            torch.cat([self.backward_context_initial[0] for b in range(B)], 1),
            torch.cat([self.backward_context_initial[1] for b in range(B)], 1)
        )
        for i in range(N - 1, -1, -1):
            next_input = autograd.Variable(torch.from_numpy(data[i, :]).float())
            out, hidden = self.backward_context_lstm(next_input.view(1, B, K), hidden)
            backward_xcribe_data.append(out)

        xcribe_data = []
        for i in range(N):
            xcribe_data.append(torch.cat([forward_xcribe_data[i], backward_xcribe_data[i]], 2))

        forward_precalc = [[None for _ in range(N)] for _ in range(N)]
        # forward_precalc[i, j, :] => [i, j]
        for i in range(N):
            hidden = (
                torch.cat([self.forward_initial[0] for b in range(B)], 1),
                torch.cat([self.forward_initial[1] for b in range(B)], 1)
            )
            for j in range(i, min(N, i + DATA_MAX_SEG_LEN)):
                next_input = xcribe_data[j]
                out, hidden = self.forward_lstm(next_input, hidden)
                forward_precalc[i][j] = out

        backward_precalc = [[None for _ in range(N)] for _ in range(N)]
        # backward_precalc[i, j, :] => [i, j]
        for i in range(N):
            hidden = (
                torch.cat([self.backward_initial[0] for b in range(B)], 1),
                torch.cat([self.backward_initial[1] for b in range(B)], 1)
            )
            for j in range(i, max(-1, i - DATA_MAX_SEG_LEN), -1):
                next_input = xcribe_data[j]
                out, hidden = self.backward_lstm(next_input, hidden)
                backward_precalc[j][i] = out
        return forward_precalc, backward_precalc

    def infer(self, data):
        N, B, K = data.shape
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
                t_val = self.W(self.Phi(self.V(seg_encoding)))
                t = t_val + log_alphas[j][2]
                # print("t_val: ", t_val)
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

def parse_file(train_filename, embedding, use_max_len=True):
    train_file = open(train_filename)
    sentences = []
    labels = []
    label = []
    POS_labels = set()
    sentence = ""
    label_sum = 0
    for line in train_file:
        if line.startswith("# text = "):
            sentence = line[9:].strip().replace(" ", "")
            N = len(sentence)
            if use_max_len:
                max_len = MAX_SENTENCE_LEN
            else:
                max_len = N
            sentence_vec = np.zeros((max_len, EMBEDDING_DIM))
            for i in range(min(N, max_len)):
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
                if len(sentence) != 0:
                    labels.append((label, sentence))
                label = []
                label_sum = 0
                sentence = ""
            else:
                if (label_sum + len(parts[1])) <= max_len:
                    label_sum += len(parts[1])
                    label.append((parts[3], len(parts[1])))

    return sentences, labels

def count_correct_labels(predicted, gold):
    correct_count = 0
    predicted_set = set()
    chars = 0
    for tag, l in predicted:
        label = (tag, chars, chars + l)
        predicted_set.add(label)
        chars += l
    chars = 0
    for tag, l in gold:
        label = (tag, chars, chars + l)
        if label in predicted_set:
            correct_count += 1
        chars += l
    return correct_count

def eval_f1(seg_rnn, pairs):
    gold_segs = 0
    predicted_segs = 0
    correct_segs = 0
    for idx, (datum, (gold_label, sentence)) in enumerate(pairs):
        if idx % 25 == 0:
            print("eval ", idx)
        predicted_label = seg_rnn.infer(datum.reshape(len(sentence), 1, EMBEDDING_DIM))
        predicted_segs += len(predicted_label)
        gold_segs += len(gold_label)
        correct_segs += count_correct_labels(predicted_label, gold_label)
    if predicted_segs > 0:
        precision = correct_segs / predicted_segs
    else:
        precision = 0.0
    print("Precision: ", precision)
    if gold_segs > 0:
        recall = correct_segs / gold_segs
    else:
        recall = 0.0
    print("Recall: ", recall)
    if precision > 0 and recall > 0:
        f1 = 2.0 / (1.0 / precision + 1.0 / recall)
    else:
        f1 = 0.0
    print("F1: " , f1)
    f = open("eval_scores.txt", "a+")
    f.write("%f %f %f\n" % (precision, recall, f1))
    f.close()

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segmental RNN.')
    parser.add_argument('--train', help='Training file')
    parser.add_argument('--test', help='Test file')
    parser.add_argument('--embed', help='Character embedding file')
    parser.add_argument('--model', help='Saved model')
    parser.add_argument('--lr', help='Learning rate (default=0.01)')
    parser.add_argument('--evalOnly', help='Evaluate this model')
    args = parser.parse_args()

    embedding = parse_embedding(args.embed)
    print("Done parsing embedding")
    data, labels = parse_file(args.train, embedding, use_max_sentence_len_training)
    pairs = list(zip(data, labels))
    # pairs = pairs[0:250]
    print("Done parsing training data")

    if args.test is not None:
        test_data, test_labels = parse_file(args.test, embedding, False)
        test_pairs = list(zip(test_data, test_labels))
        print("Done parsing testing data")

    if args.model is not None:
        seg_rnn = torch.load(args.model)
    else:
        seg_rnn = SegRNN()

    if args.lr is not None:
        learning_rate = float(args.lr)
    else:
        learning_rate = 0.01

    optimizer = torch.optim.Adam(seg_rnn.parameters(), lr=learning_rate)
    count = 0.0
    sum_loss = 0.0
    correct_count = 0.0
    sum_gold = 0.0
    sum_pred = 0.0
    for batch_num in range(1000):
        random.shuffle(pairs)
        for i in range(0, min(BATCH_SIZE, len(pairs)), MINIBATCH_SIZE):
            seg_rnn.train()
            start_time = time.time()

            optimizer.zero_grad()

            
            if use_max_sentence_len_training:
                max_len = MAX_SENTENCE_LEN
                batch_size = min(MINIBATCH_SIZE, len(pairs) - i)
            else:
                max_len = len(pairs[i][1][1])
                batch_size = 1
            batch_data = np.zeros((max_len, batch_size, EMBEDDING_DIM))
            batch_labels = []
            for idx, (datum, (label, sentence)) in enumerate(pairs[i:i+batch_size]):
                batch_data[:, idx, :] = datum
                batch_labels.append(label)
            loss = seg_rnn.calc_loss(batch_data, batch_labels)
            sum_loss += loss.data[0]
            count += 1.0 * batch_size
            loss.backward()

            optimizer.step()

            seg_rnn.eval()
            print("Batch ", batch_num, " datapoint ", i, " avg loss ", sum_loss / count)
            sentence_len = len(pairs[i][1][1])
            pred = seg_rnn.infer(batch_data[0:sentence_len, 0, np.newaxis, :])
            gold = pairs[i][1][0]
            print(pred)
            print(gold)
            print(pairs[i][1][1])
            sentence_unk = ""
            for c in pairs[i][1][1]:
                sentence_unk += c if c in embedding or c in "0123456789" else "_"
            print(sentence_unk)
            correct_count += count_correct_labels(pred, gold)
            sum_gold += len(gold)
            sum_pred += len(pred)
            cum_prec = correct_count / sum_pred
            cum_rec = correct_count / sum_gold
            if cum_prec > 0 and cum_rec > 0:
                print("F1: ", 2.0 / (1.0 / cum_prec + 1.0 / cum_rec)," cum. precision: ", cum_prec, " cum. recall: ", cum_rec)
            # print(seg_rnn.Y_encoding[0], seg_rnn.Y_encoding[5])
            # print(seg_rnn.Y_encoding[0].grad, seg_rnn.Y_encoding[5].grad)
            #for param in seg_rnn.parameters():
            #    print(param)

            end_time = time.time()
            print("Took ", end_time - start_time, " to run ", MINIBATCH_SIZE, " training sentences")

        if args.test is not None:
            torch.save(seg_rnn, "seg_rnn_deep_" + str(batch_num) + ".pt")
            if (batch_num + 1) % 30 == 0:
                eval_f1(seg_rnn, test_pairs)
