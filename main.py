import nltk.stem
import csv
import scipy as sp
from sklearn.feature_extraction.text import CountVectorizer
import numpy


# this function return normalized distance betwee two sparse tables
def dist_norm(v1, v2):
    v1_normalized = v1 / sp.linalg.norm(v1.toarray());
    v2_normalized = v2 / sp.linalg.norm(v2.toarray());
    delta = v1_normalized - v2_normalized;
    return sp.linalg.norm(delta.toarray());


raw_data = []
with open("train.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        raw_data.append(row)

raw_words = []  # used to fit stemmed vectorizer
raw_pairs = []  # used to store each pair of questions in train data
answers = []  # right is_duplicate train answers
for i in range(1, len(raw_data)):
    raw_words.append(raw_data[i][3])
    raw_words.append(raw_data[i][4])
    raw_pairs.append(raw_data[i][3:5])
    answers.append(raw_data[i][5])

english_stemmer = nltk.stem.SnowballStemmer("english")
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


vectorizer = StemmedCountVectorizer(min_df=1, stop_words="english")
vectorizer.fit(raw_words)


def vect(arr):
    new_arr = [arr]
    return vectorizer.transform(new_arr);


pairs = []  # used to store vectorized train questions
for row in raw_pairs:
    new_row = []
    for sentence in row:
        new_row.append(vect(sentence))
    pairs.append(new_row)

print(len(pairs))  # some logs to know, where i am

good_pos = []
good_neg = []
for i in range(0, len(pairs)):
    ans = dist_norm(pairs[i][0], pairs[i][1]);
    if answers[i] == '1':
        good_pos.append(ans);
    else:
        good_neg.append(ans);
print(sum(good_pos) / len(good_pos));
print(sum(good_neg) / len(good_neg));
lim = (sum(good_pos) / len(good_pos) + sum(good_neg) / len(good_neg)) / 2;
# the idea of the code above is to ged average distance betwee different questions and between similar questiond
# the median of two distances is how we check questions for duplicates

test_answers = [];  # used to store predictions for test data

print("we are fine over here");  # another log telling that we've reached a certainline of code
with open("test.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  # change contents to floats
    for row in reader:  # each row is a list
        new_dist = dist_norm(vect(row[1]), vect(row[2]));  # comparing vectorization of two answers
        new_ans = '0';
        if (new_dist < lim):
            new_ans = '1';
        test_answers.append(new_ans);

with open('answers.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['test_id', 'is_duplicate']);
    for i in range(1, len(test_answers)):
        writer.writerow([i - 1, int(test_answers[i])]);