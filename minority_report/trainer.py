from minority_report.input import Input
from sklearn.model_selection import train_test_split
class Trainer:
    def __init__(self,X, y):
        self.X = None
        self.y = None

    def load_data_from_input_class(self, number_of_observations, x_length, y_length):
        self.X, self.y = Input().combining_load_data_and_X_y(number_of_observations, x_length, y_length)
        return self.X, self.y

    def holdout(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.2)
        return X_train, X_test, y_train, y_test

    def init_model(x_length, y_length, lat_size, lon_size):
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
        return model


if __name__ == '__main__':
    number_of_observations = 50 #50 observations
    x_length = 24 #24h avant
    y_length = 3 #3h apres
    print('1. Getting X, y from instanciating Trainer class ')
    X, y = Trainer().load_data_from_input_class(number_of_observations, x_length, y_length))
    print('2. Train test split')
    holdout()
    print('2. Init model')
    # to recuprere from coords_to_matrix
    init_model(x_length, y_length, lat_size, lon_size)


  # holdout method:train_test_split ici or in Training class
  # init model
  # fit model
  # predict
