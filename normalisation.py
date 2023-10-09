import logging

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
class Normalisation:
    def normalize_data(self, scaler:str, mySeries, field:str, namesofMySeries):
        """
           This functions normalises the data based on the input scaler method
           and save the results both into  mySeries list,
           and into another list named Series_arrats as arrays
           Arguments:
             scaler: normalization method
             mySeries: list of df data
             field: Column to apply the normalisation
           Returns:
             mySeries: list of series
             Series_arrays: list of arrays
         """
        Series_arrays = {}
        for i in range(len(mySeries)):
            if scaler == 'minmax':
                norm_scaler = MinMaxScaler()
            elif scaler == 'standard':
                norm_scaler = StandardScaler()
            elif scaler == 'robust':
                norm_scaler = RobustScaler()
            else:
                norm_scaler = MinMaxScaler()
            key = namesofMySeries[i]
            Series_arrays[key] = mySeries[i]
            mySeries[i] = norm_scaler.fit_transform(mySeries[i][['Temperature']])
            Series_arrays[key]['Normalised'] = mySeries[i]
            mySeries[i] = mySeries[i].reshape(len(mySeries[i]))
        logging.info('Normalisation finished')
        return mySeries, Series_arrays