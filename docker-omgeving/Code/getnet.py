# Basic imports
import lightnet as ln

# Settings
ln.logger.setConsoleLevel('ERROR')  # Only show error log messages

def getnet(networkname):
    if networkname == 'Yolo':
        return ln.models.Yolo()
    elif networkname == 'Yolt':
        return ln.models.Yolt()
    elif networkname == 'DYolo':
        return ln.models.DYolt()
    elif networkname == 'TinyYolo':
        return ln.models.TinyYolo()
    elif networkname == 'MobileNetYolo':
        return ln.models.MobileNetYolo()
    else:
        print('An unsupported network was chosen, exiting.')
        quit()
