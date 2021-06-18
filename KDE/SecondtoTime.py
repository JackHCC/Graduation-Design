def sec2time(sec):
    ''' Convert seconds to '#D days#, HH:MM:SS.FFF' '''
    if hasattr(sec,'__len__'):
        return [sec2time(s) for s in sec]
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    #d, h = divmod(h, 24)
    pattern = r'%02d:%02d:%02d'
    return pattern % (h, m, s)
