
import os
import pandas as pd
import pickle

class Input:
    # passer de nos map à notre liste de tensors d'entrainement

    def __init__(self):
        self.data = None

    def load_data(self):
        root_dir = os.path.dirname(os.path.dirname(__file__))
        pickle_path = os.path.join(root_dir, 'raw_data', 'clean.pickle')
        with open(pickle_path, 'rb') as f:
            df = pickle.load(f)
        self.data = df
        return self.data

    def get_observation_target(self,img3D_conv):
        position = np.random.randint(0,img3D_conv.shape[0]-27)
        observation = img3D_conv[position:position+24]
        target = img3D_conv[position+24:position+27].flatten() # include as function parameter
        return observation, target

    def get_X_y(self,img3D_conv, number_of_observations):
        X = []
        y = []
        for n in range(number_of_observations):
            X_subsample, y_subsample = get_observation_target(img3D_conv)
            X.append(X_subsample)
            y.append(y_subsample)
        X = np.array(X)
        y = np.array(y)
        return X, y


if __name__ == '__main__':
    input = Input()
    input.load_data()
    X, y = input.get_X_y(img3D_conv, 50)
    X.shape
    y.shape
