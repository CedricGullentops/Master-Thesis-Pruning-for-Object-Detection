# Basic imports
import lightnet as ln

# Settings
ln.logger.setConsoleLevel('ERROR')  # Only show error log messages

def softprune(model, *argv):
    print('Using l2prune technique')
    #if self.manner == 'hard':
    #    hardprune(self.model)
    #elif self.manner == 'soft':
    #    softprune(self.model)
    #else:
    #    combination(self.model)
    #return


    print (model.parameters)

    model = ln.models.Yolo()
    print(type(model.layers[0][0]))
    for parameter in model.parameters():
        print('new parameter')
        print(parameter.data.size())
        #print(parameter.data)
    print(model.layers[0][0].out_channels)

    for arg in argv:
        print('new arg:')
        # print list(model.parameters())[0].data.numpy()
    return
