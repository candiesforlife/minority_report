from minority_report.input import Input
from minority_report.matrix import Matrix

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import models
from tensorflow.keras import layers

class Trainer:
    def __init__(self):
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = None
        self.y_pred = None


    def load_data_from_input_class(self, number_of_observations, x_length, y_length):
        self.X, self.y = Input().combining_load_data_and_X_y(number_of_observations, x_length, y_length)
        return self.X, self.y

    def holdout(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.2)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def init_model(self,x_length, y_length, lat_size, lon_size):
        print('initializing model')
        model = models.Sequential()
        print('adding conv2D 1')
        model.add(layers.Conv2D(16, kernel_size = 5, activation = 'relu',padding='same',
                                input_shape = (x_length, lat_size, lon_size),
                               data_format='channels_first'))
        model.add(layers.MaxPooling2D(2, data_format='channels_first'))
        print('adding conv2D 2')
        model.add(layers.Conv2D(128, kernel_size = 3, activation = 'relu', padding='same',  data_format='channels_first'))
        model.add(layers.MaxPooling2D(2, data_format='channels_first'))
        print('adding conv2D 3')
        model.add(layers.Conv2D(64, kernel_size = 3, activation = 'relu', padding='same', data_format='channels_first' ))
        model.add(layers.MaxPooling2D(2, data_format='channels_first'))
        print('flattening')
        model.add(layers.Flatten())
        print('adding dense layer 1')
        model.add(layers.Dense(50, activation = 'relu'))
        print('adding dense layer 2')
        model.add(layers.Dense(500, activation = 'relu'))
        #print('adding dense layer 2')
        #model.add(layers.Dropout(rate=0.5))
        print('adding dense layer 3')
        model.add(layers.Dense(y_length * lat_size * lon_size, activation = 'relu'))
        print('Reshaping')
        model.add(layers.Reshape((y_length, lat_size, lon_size)))
        print('compiling')
        model.compile(loss = 'mse',
                      optimizer = 'adam',
                      metrics = 'mae')
        print('Done !')
        self.model = model
        return self.model

    def fit_model(self,batch_size, epochs, patience):
        es = EarlyStopping(patience = patience, restore_best_weights=True)
        self.model.fit(self.X_train, self.y_train,
                      batch_size = batch_size,
                      epochs = epochs,
                      validation_split = 0.3,
                      callbacks = es)
        return self.model

    def evaluate_model(self):
        result = self.model.evaluate(self.X_test, self.y_test)
        return result

    def predict_model(self):
        self.y_pred = self.model.predict(self.X_test)
        return self.y_pred

    def save_y_pred_to_pickle(self):
        '''
        Saves to  y_pred pickler
        '''
        root_dir = os.path.dirname(os.path.dirname(__file__))
        pickle_path = os.path.join(root_dir, 'raw_data', 'y_pred.pickle')

        with open(pickle_path, 'wb') as f:
            pickle.dump(self.y_pred, f)



    def training_model(self, number_of_observations, x_length, y_length, lat_size, lon_size, batch_size, epochs, patience):
        print('7. Getting X, y from instanciating Trainer class ')
        self.load_data_from_input_class(number_of_observations, x_length, y_length)
        print('10. Train test split')
        self.holdout()
        print('11. Init model')
        self.init_model(x_length, y_length, lat_size, lon_size)
        print('12. Fit model')
        self.fit_model(batch_size, epochs, patience)
        print('13. Evaluate')
        self.evaluate_model()
        # print('14. Predict')
        # self.predict_model()
        # print('15. Save y_pred to pickle')
        # self.save_y_pred_to_pickle()
        return self



if __name__ == '__main__':
    print('1. Creating an instance of Matrix class')
    df = Matrix()
    lat_size, lon_size,img3D_conv = df.crime_to_img3D_con()
    print('6. Saving image filtered 3d convoluted to pickle')
    df.save_data()
    x_length = 24 #24h avant
    y_length = 3 #3h apres
    number_of_observations = 50 #50 observations
    batch_size = 32
    epochs = 100
    patience = 5
    trainer = Trainer()
    trainer.training_model(number_of_observations, x_length, y_length, lat_size, lon_size, batch_size, epochs, patience)


