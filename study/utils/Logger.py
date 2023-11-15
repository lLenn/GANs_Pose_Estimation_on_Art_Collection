import os
import logging
import time
from pathlib import Path
 
class Logger:
    def __init__(self, output, phase="train"):
        output_path = Path(output)
        if not output_path.exists():
            print('=> creating {}'.format(output_path))
            output_path.mkdir()
            
        logging.basicConfig(
            filename=os.path.join(output, f"{phase}_{time.strftime('%Y-%m-%d-%H-%M')}.log"),
            format='%(asctime)-15s %(message)s'
        )
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        
        console = logging.StreamHandler()
        logging.getLogger('').addHandler(console)
        
    def info(self, *arg, **kwargs):
        self.logger.info(*arg, **kwargs)