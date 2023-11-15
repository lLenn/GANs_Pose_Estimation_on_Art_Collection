import argparse

class ValidateNetworkProgram:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Test keypoints network")
        self.parser.add_argument(
            "--cfg",
            help="experiment configure file name",
            required=True,
            type=str
        )
        self.parser.add_argument(
            "opts",
            help="Modify config options using the command-line",
            default=None,
            nargs=argparse.REMAINDER,
        )
        self.parser.add_argument(
            "--world_size",
            help="Modify config options using the command-line",
            type=int,
            default=8,
        )
        self.args = self.parser.parse_args()

    def getArgument(self, name):
        return getattr(self.args, name)
        