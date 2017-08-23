from time import gmtime, strftime


def get_time():
    return strftime("[%Y-%m-%d %H:%M:%S]", gmtime())