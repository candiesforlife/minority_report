class Input:

    #Dans la class model, on recupere list of tensors
    # method qui definit kernel
    # method qui fait passer le kernel dans chaque tensor
    # method X Y

    def __init__:
        #model
        #X_train is our list of tensors

    def get_observation_target(img3D_conv):
        position = np.random.randint(0,img3D_conv.shape[0]-27)
        observation = img3D_conv[position:position+24]
        target = img3D_conv[position+24:position+27]
        return observation, target


    def get_X_y(img3D_conv, number_of_observations):
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
    # X, y = get_X_y(img3D_conv, 50 )
    # X.shape
    # y.shape
