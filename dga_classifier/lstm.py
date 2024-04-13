"""Train and test LSTM classifier"""
import dga_classifier.data as data
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from tensorflow.keras.models import save_model
import sklearn
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split


def build_model(max_features, maxlen):
    """Build LSTM model"""
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop')

    return model

def run(max_epoch=25, nfolds=10, batch_size=128):
    """Run train/test on logistic regression model"""
    indata = data.get_data() #获得了相同数量的恶意域名和良性域名
    X,labels = [], []
    for x in indata:
        X.append(x[1])
        labels.append(x[0])
    # Extract data and labels
    
    # Generate a dictionary of valid characters
    valid_chars = {x:idx+1 for idx, x in enumerate(set(''.join(X)))}
    with open("encoding.txt", "w") as f:
        f.write(str(valid_chars))
    print(valid_chars)
    max_features = len(valid_chars) + 1
    maxlen = np.max([len(x) for x in X])
    print(f" max_features = {max_features}, maxlen = {maxlen}")

    # Convert characters to int and pad
    X = [[valid_chars[y] for y in x] for x in X] #将文本序列转换为整数序列
    # print(X)
    X = sequence.pad_sequences(X, maxlen=maxlen) #保持整数序列的长度一致

    # Convert labels to 0-1
    y = [0 if x == 'benign' else 1 for x in labels] #标签 二分类
    final_data = []

    for fold in range(nfolds):
        print ("fold %u/%u" % (fold+1, nfolds)) #当前折数
        X_train, X_test, y_train, y_test, _, label_test = train_test_split(X, y, labels, test_size=0.2)

        print ('Build model...')
        model = build_model(max_features, maxlen)
        
        print ("Train...")
        X_train, X_holdout, y_train, y_holdout = train_test_split(X_train, y_train, test_size=0.05) #将训练集的数据再分
        best_iter = -1
        best_auc = 0.0
        out_data = {}

        for ep in range(max_epoch):
            model.fit(np.array(X_train), np.array(y_train), batch_size=batch_size, epochs=1, shuffle=False)

            # t_probs = model.predict_proba(X_holdout) #预测这一折训练集的预测概率
            t_probs = model.predict(X_holdout) #预测这一折训练集的预测概率
            t_auc = sklearn.metrics.roc_auc_score(y_holdout, t_probs) #用ROC衡量二分类模型的预测性能

            print ('Epoch %d: auc = %f (best=%f)' % (ep, t_auc, best_auc))

            if t_auc > best_auc: #如果这一折的性能比best_auc好
                best_auc = t_auc
                best_iter = ep #记录当前epoch
                
                probs = model.predict_proba(X_test) #在测试集上的预测概率

                out_data = {'y':y_test, 'labels': label_test, 'probs':probs, 'epochs': ep,
                            'confusion_matrix': sklearn.metrics.confusion_matrix(y_test, probs > .5)}

                print (sklearn.metrics.confusion_matrix(y_test, probs > .5))
            else: #如果没有
                # No longer improving...break and calc statistics
                if (ep-best_iter) > 2: #超过两次epoch的ROC性能都没有之前的好，则结束
                    break     
                
        evaulation= model.evaluate(np.array(X_test), np.array(y_test))
        # print("Test Loss:", loss)
        print("Test evaulation:", evaulation)       
    
        final_data.append(out_data) #记录了每一折的评估效果和模型的预测效果
        model.save("lstm_model1.h5")  
        print("Model saved successfully!")
        
    return final_data
