import logging

logging.basicConfig(
    level=logging.INFO,
    filename='log/stock_analysis.log',
    filemode='w',
    format='Date-Time :: %(asctime)s ::  %(levelname)s :: %(funcName)s :: %(lineno)d :: [%(levelname)s] (%('
           'threadName)-10s) %(message)s')


def notify(message):
    """
    Method that saves in the file.log the message that is passed by parameter
    :param message: Message I want to save in the .log
    """
    logging.info(str(message) + "\n\n")
