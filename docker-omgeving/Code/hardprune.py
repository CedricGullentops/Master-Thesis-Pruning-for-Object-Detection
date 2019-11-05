# Basic imports
import lightnet as ln

# Settings
ln.logger.setConsoleLevel('ERROR')  # Only show error log messages

def hardprune(network, *argv):
    for arg in argv:
        print list(network.parameters())[0].data.numpy()
    return
