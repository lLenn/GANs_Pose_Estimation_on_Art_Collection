from mmengine import Config

class ViTPoseConfig():
    def create(file, options=dict()):
        config = Config.fromfile(file)
        config.merge_from_dict(options)
        return config