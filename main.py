from stopwords import stopword
from utils import *
from sklearn import svm
if __name__ == '__main__':
    stopword = stopword()
    X_train, X_test, y_train, y_test = read_data()
    df_train = pd.DataFrame(X_train, columns=['title'])
    df_test = pd.DataFrame(X_test, columns=['title'])
    preprocessed_train = preprocessing(df_train, stopword)
    preprocessed_test = preprocessing(df_test, stopword)

    X_train, X_test, vector = vectorizor(preprocessed_train, preprocessed_test)
    result = train(X_train, y_train, svm.LinearSVC())
    print(test(result, X_test, y_test))
    
    test = pd.read_csv('./data/test_data.csv')
    test_data = preprocessing(test, stopword)
    res_test = vector.transform(test_data)
    pred = predict(result, res_test)
    print(pred)
