import keras
import sys
import h5py
import numpy as np

clean_data_filename = str(sys.argv[1])
poisoned_data_filename = str(sys.argv[2])
model_filename = str(sys.argv[3])
model_prime_filename = str(sys.argv[4])

def Good_Net(model, model_prime, data, N):
    """
    GoodNet
    """
    x = data
    y_pred = np.argmax(model.predict(x), axis=1)
    y_pred_prime = np.argmax(model_prime.predict(x), axis=1)
    y_hat = [y_pred[i] if y_pred[i] == y_pred_prime[i] else N for i in range(len(y_pred))]
    
    return y_hat

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data

def main():
    cl_x_test, cl_y_test = data_loader(clean_data_filename)
    bd_x_test, bd_y_test = data_loader(poisoned_data_filename)

    bd_model = keras.models.load_model(model_filename)
    bd_model_prime = keras.models.load_model(model_prime_filename)

    N = len(set(cl_y_test))

    cl_label_p = Good_Net(bd_model, bd_model_prime, cl_x_test, N)
    clean_accuracy = np.mean(np.equal(cl_label_p, cl_y_test))*100
    print('Clean Classification Accuracy:', clean_accuracy)
    
    bd_label_p = Good_Net(bd_model, bd_model_prime, bd_x_test, N)
    asr = np.mean(np.equal(bd_label_p, bd_y_test))*100
    print('Attack Success Rate:', asr)

if __name__ == '__main__':
    main()
