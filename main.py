import numpy as np

#needed
def distance(thing1, thing2):
    return np.linalg.norm(thing1 - thing2)

class BinaryClassifier:
    def __init__(self, margin=0.1):
        self.margin = margin
        self.pos_set = []

        self.neg_set = []
        self.hypercube = None 

    def fit(self, X, y):
        # only taking positive stuff
        for i in range(len(X)):
            if y[i] == 1:
                self.pos_set.append(X[i])
            else:
                self.neg_set.append(X[i])
        
        self._make_box_thing()

    def _make_box_thing(self):
        if not self.pos_set:
            return 

     
        mins = np.min(np.array(self.pos_set), axis=0) - self.margin
        maxs = np.max(np.array(self.pos_set), axis=0) + self.margin
        self.hypercube = (mins, maxs)

        # box touches any neg, squish it
        for neg in self.neg_set:
            if self._inside_box(neg):
                shift = 0.01  # arbitrary tiny nudge
                mins = np.minimum(mins, neg - shift)
                maxs = np.maximum(maxs, neg + shift)
        self.hypercube = (mins, maxs) 

    def _inside_box(self, x):
        mins, maxs = self.hypercube
        return np.all(x >= mins) and np.all(x <= maxs)

    def predict(self, X):
        
        try:
            return np.array([1 if self._inside_box(x) else 0 for x in X])
        except:
            return np.zeros(len(X))  # fallback

    def score(self, X, y):
        pred = self.predict(X)
        return np.sum(pred == y) / len(y)


#test
if __name__ == "__main__":
    # dummy 
    np.random.seed(42)
    pos = np.random.randn(20, 2) + 2  # positive cloud
    neg = np.random.randn(20, 2) - 2  # negative cloud

    X = np.vstack((pos, neg))
    y = np.array([1]*20 + [0]*20)

    clf = BinaryClassifier()
    clf.fit(X, y)
    print("Score:", clf.score(X, y))
