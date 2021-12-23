
from autocomplete import AutoComplete


class NGramAutoComplete(AutoComplete):
    def __init__(
        self,
        args,
    ):
        '''
        Set from_file = "./models_dir/" if the model was saved in the models_dir directory before.
        '''
        super().__init__()

    def load(self, dir_path: str = './models_dir/'):
        raise NotImplementedError

    def save(self, dir_path: str = './models_dir/'):
        raise NotImplementedError

    def complete(self, sent: str):
        raise NotImplementedError

    def topk(self, sent: str, k: int = 10):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
