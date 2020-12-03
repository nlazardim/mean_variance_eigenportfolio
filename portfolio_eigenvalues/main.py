from analisis import StockAnalisys


def run_model(lista_ticker, period, training_test):
    data = StockAnalisys(list_ticker=lista_ticker, data_period=period, training_period=training_test)
    data.get_data()
    data.covariance_matrix()
    data.ploting_eigenportfolio()
    data.samples()
    data.ploting_samples()


run_model(lista_ticker=['GGAL.BA', 'BMA.BA', 'SUPV.BA', 'PAMP.BA', 'YPFD.BA', 'BMA.BA', 'ALUA.BA'],
          period='W',
          training_test=30)