import random

class SomeKindOfModel(object):
    def train(self, x, y):
        print("Train Start")
        for x_i, y_i in  zip(x, y):
            print("X", x_i, "Y", y_i)
        print("Train Finish")

    def test(self, x, y):
        print("Test Start")
        for x_i, y_i in  zip(x, y):
            print("X", x_i, "Y", y_i)
        print("Test Finish")
        return 1 - (1 / len(x))


def split_data(data, prob):
    results = [], []
    for row in data:
        results[0 if random.random() < prob else 1].append(row)
    return results

def train_test_split(x, y, test_pct):
    data = zip(x, y)
    train, test = split_data(data, 1 - test_pct)
    x_train, y_train = zip(*train)
    x_test, y_test = zip(*test)
    return x_train, y_train, x_test, y_test


def run_train_and_test():
    xs = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    ys = [1, 4, 9, 16, 25, 36, 49, 64, 81]
    model = SomeKindOfModel()
    x_train, y_train, x_test, y_test = train_test_split(xs, ys, 0.33)
    model.train(x_train, y_train)
    performance = model.test(x_test, y_test)
    print("Pesformance of model", performance)

def accuracy(tp, fp, fn, tn):
    """ True positive, False positive, False negative, True negative """
    correct = tp + tn
    total = tp + fp + fn + tn
    return correct / total

def precision(tp, fp, fn, tn):
    return tp / (tp + fp)

def recall(tp, fp, fn, tn):
    return tp / (tp + fn)

def f1_score(tp, fp, fn, tn):
    p = precision(tp, fp, fn, tn)
    r = recall(tp, fp, fn, tn)
    return 2 * p * r / (p + r)

def run_proccess_score():
    in_data = [70, 4930, 13930, 981070]
    a = accuracy(*in_data)
    p = precision(*in_data)
    r = recall(*in_data)
    f1 = f1_score(*in_data)
    print("Accuracy:", a)
    print("Precision:", p)
    print("Recall:", r)
    print("F1 SCORE:", f1)

if __name__ == "__main__":
    run_train_and_test()
    run_proccess_score()

