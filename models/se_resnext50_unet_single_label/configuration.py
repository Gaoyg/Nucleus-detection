from common import *
import configparser

class Configuration(object):

    def __init__(self):
        super(Configuration, self).__init__()
        self.version='configuration version \'xxx-kaggle\''

        #features
        self.scales = [2,  4,  8, 16]

    #-------------------------------------------------------------------------------------------------------
    def __repr__(self):
        d = self.__dict__.copy()
        str=''
        for k, v in d.items():
            str +=   '%32s = %s\n' % (k,v)

        return str


    def save(self, file):
        d = self.__dict__.copy()
        config = configparser.ConfigParser()
        config['all'] = d
        with open(file, 'w') as f:
            config.write(f)


    def load(self, file):
        # config = configparser.ConfigParser()
        # config.read(file)
        #
        # d = config['all']
        # self.num_classes     = eval(d['num_classes'])
        # self.multi_num_heads = eval(d['multi_num_heads'])
        raise NotImplementedError