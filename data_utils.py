from sklearn import preprocessing


class normaliser:
    def __init__(self, train_data):
        self.scaler = preprocessing.StandardScaler().fit(train_data['X'])

    def __call__(self, data):
        '''expect data as a dict with 'X', 'y' keys'''
        data['X'] = self.scaler.transform(data['X'])
        return data
