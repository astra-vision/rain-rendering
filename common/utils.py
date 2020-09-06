import sys

def printError(msg):
    print('\n\x1b[2;30;41m[ERROR]\x1b[0m  %s' % msg)

def printSuccess(msg):
    print('\n\x1b[2;30;42m[SUCCESS]\x1b[0m  %s' % msg)

def printWarning(msg):
    print('\x1b[2;30;43m[WARNING]\x1b[0m  %s' % msg)

def printProgressBar (iteration, total, prefix = 'Progress:', suffix = 'Complete', decimals = 2, length = 100, fill = '#'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    sys.stdout.flush()
    # Print New Line on Complete
    if iteration == total:
        print()
