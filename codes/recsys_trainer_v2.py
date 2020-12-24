
from transformers import (Trainer)
from torch.utils.data.dataloader import DataLoader
from enum import Enum

class DatasetType(Enum):
    train = "Train"
    valid = "Valid"
    test = "Test"


class RecSysTrainer(Trainer):
    
    def __init__(self, *args, **kwargs):
        self.global_step = 0

        '''
        if 'fast_test' not in kwargs:
            self.fast_test = False
        else:
            self.fast_test = kwargs.pop('fast_test')
        '''

        if 'log_predictions' not in kwargs:
            self.log_predictions = False
        else:
            self.log_predictions = kwargs.pop('log_predictions')

        #self.create_metrics()
        
        super(RecSysTrainer, self).__init__(*args, **kwargs)

    def get_train_dataloader(self) -> DataLoader:
        return self.train_dataloader            
        
    def get_eval_dataloader(self) -> DataLoader:
        return self.eval_dataloader

    def get_test_dataloader(self) -> DataLoader:
        return self.test_dataloader

    def set_train_dataloader(self, dataloader):
        self.train_dataloader = dataloader
        
    def set_eval_dataloader(self, dataloader):
        self.eval_dataloader = dataloader

    def set_test_dataloader(self, dataloader):
        self.test_dataloader = dataloader

    def num_examples(self, dataloader):
        return len(dataloader)

    #Finish training fix. Will probably have to extend predict_loop(), maybe evaluate() and predict() and also compute_metrics