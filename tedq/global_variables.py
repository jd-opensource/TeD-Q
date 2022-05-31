'''
global variable for all files
'''
class GlobalVariables():
    '''
    global variable for all files
    '''

    _global_dict = {}


    @classmethod
    def set_value(cls, key, value):
        '''
        set value
        '''
        cls._global_dict[key] = value

    @classmethod
    def get_value(cls, key, default_value=None):
        """获得一个全局变量,不存在则返回默认值"""
        try:
            return cls._global_dict[key]
        except KeyError:
            return default_value
