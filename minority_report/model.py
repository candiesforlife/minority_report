class Model:


    #Dans la class model, on recupere list of tensors
    # method qui definit kernel
    # method qui fait passer le kernel dans chaque tensor
    # method X Y

    def __init__:
        #model
        #X_train is our list of tensors

    def get_observation_target(X_train):
        position = np.random.randint(0,X_train.shape[0]-24)
        observation = X_train[position:position+24]
        target = X_train[position+24:position+27]
        return observation, target

    def get_X_y(X_train, number_of_observations):
        X = []
        y = []
        for n in range(number_of_observations):
            X_subsample, y_subsample = get_observation_target(X_train)
            X.append(X_subsample)
            y.append(y_subsample)
        X = np.array(X)
        y = np.array(y)
        return X, y
