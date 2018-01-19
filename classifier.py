from sklearn import svm as sksvm

def svm(train_features, train_label):
    model = sksvm.SVC(kernel = "linear").fit(train_features,train_label)
    return model

# More model insert here >:)

def decision_function(model):
    return model.decision_function()

def predict(model, test_features):
    return model.predict(test_features)
