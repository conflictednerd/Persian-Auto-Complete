
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

        # so that multi-processing with cuda is possible
        set_start_method('spawn')

        self.MODEL_NAME = 'ngram'
        self.MODEL_DIR = args.models_dir
        self.TRAIN_DATA_PATH = args.train_data_path
        self.CLEANIFY = args.cleanify
        self.DEVICE = args.device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        # self.config = AutoConfig.from_pretrained(
        #     self.MODEL_DIR if args.load else self.MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_DIR if args.load else self.MODEL_NAME)
        # # TODO: .to(device) ? is it enough? how will loading and saving using different devices affect saving and loading
        # self.model = AutoModelForMaskedLM.from_pretrained(
        #     self.MODEL_DIR if args.load else self.MODEL_NAME).to(self.DEVICE)

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
