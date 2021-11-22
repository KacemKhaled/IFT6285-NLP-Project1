from datetime import datetime
from typing import Optional

import os
import datasets
import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.functional import accuracy, precision, recall, f1, average_precision
from torchmetrics.functional import confusion_matrix, matthews_corrcoef
import torch.nn.functional as F
from pytorch_lightning.callbacks import LearningRateMonitor
from sklearn.metrics import classification_report, recall_score, precision_score
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.metrics import matthews_corrcoef as mcc
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import classification_report


from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from transformers import RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification

BATCH_SIZE = 16 #32




from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)



class GLUEDataModule(LightningDataModule):

    task_text_field_map = {
        "cola": ["sentence"],
        "sst2": ["sentence"],
        "mrpc": ["sentence1", "sentence2"],
        "qqp": ["question1", "question2"],
    }

    glue_task_num_labels = {
        "cola": 2,
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
    }

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        model_name_or_path: str,
        task_name: str = "mrpc",
        max_seq_length: int = 128,
        train_batch_size: int = BATCH_SIZE,
        eval_batch_size: int = BATCH_SIZE,
        num_workers: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers

        self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.glue_task_num_labels[task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage: str):
        self.dataset = datasets.load_dataset("glue", self.task_name)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]
        print(self.dataset)
        print(self.eval_splits)


    def prepare_data(self):
        datasets.load_dataset("glue", self.task_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size,num_workers=self.num_workers)

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size,num_workers=self.num_workers)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size,num_workers=self.num_workers) for x in self.eval_splits]

    def test_dataloader(self):
            print(self.dataset["test"])
        # if len(self.eval_splits) == 1:
            print('test data loader')
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size,num_workers=self.num_workers)
        # elif len(self.eval_splits) > 1:
        #     return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size,num_workers=self.num_workers) for x in self.eval_splits]

    # def predict_dataloader(self):
    #     manual_data = {

    #     }
    #     self.dataset = datasets.load_dataset("glue", self.task_name)

    #     for split in self.dataset.keys():
    #         self.dataset[split] = self.dataset[split].map(
    #             self.convert_to_features,
    #             batched=True,
    #             remove_columns=["label"],
    #         )
    #         self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
    #         self.dataset[split].set_format(type="torch", columns=self.columns)

        
    #     return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size,num_workers=self.num_workers)

    def convert_to_features(self, example_batch, indices=None):

        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs, max_length=self.max_seq_length, pad_to_max_length=True, truncation=True
        )

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch["label"]

        return features


def print_params(model):
    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())

    print('The BERT model has {:} different named parameters.\n'.format(len(params)))

    print('==== Embedding Layer ====\n')

    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== First Transformer ====\n')

    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== Output Layer ====\n')

    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


class GLUETransformer(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int = 2,
        task_name: str = "mrpc",
        learning_rate: float = 5e-5,#0.05,#2e-5, #best of 5e-5, 3e-5, 2e-5 , 6e-5
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = BATCH_SIZE,#32,
        eval_batch_size: int = BATCH_SIZE,
        eval_splits: Optional[list] = None,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels) #AutoConfig
        # AutoModelForSequenceClassification
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, 
            config=self.config,
            #num_labels = 2, # The number of output labels--2 for binary classification.
            #output_attentions = False, # Whether the model returns attentions weights.
            #output_hidden_states = False, # Whether the model returns all hidden-states.
            )
        print_params(self.model)
        self.metric = datasets.load_metric(
            "glue", self.hparams.task_name, experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )

    def forward(self, **inputs):
        print('----------------------------')
        print(inputs)
        print('----------------------------')
        # outputs = self.model(**inputs)
        # out =  outputs[1]
        # return F.log_softmax(out, dim=1)
        return self.model(**inputs)
    # def forward(self, x):
    #     out = self.model(x)
    #     return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        y = batch['labels']
        outputs = self(**batch)
        loss, logits =  outputs[:2]
        if self.hparams.num_labels >= 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()
        #loss = F.nll_loss(logits, y)

        # x, y = batch
        # logits = self(x)
        # loss = F.nll_loss(logits, y)
        # preds = torch.argmax(logits, dim=1)
        
        # precision, recall, f1, average_precision, confusion_matrix
        acc = accuracy(preds, y)

        # self.log("train_loss", loss, on_epoch=True) # on_step=True, on_epoch=False
        # self.log("train_acc", acc, on_epoch=True) # on_step=True, on_epoch=False

        metrics = {'train_loss':loss, 'train_acc':acc}
        self.log_dict(metrics,on_epoch=True ,prog_bar=True)
        return loss

    def evaluate(self, batch, stage=None):
        
        y = batch['labels']
        outputs = self(**batch)
        loss, logits =  outputs[:2]
        if self.hparams.num_labels >= 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()
        # x, y = batch
        # logits = self(x)
        # loss = F.nll_loss(logits, y)
        # preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            # self.log(f"{stage}_loss", loss, prog_bar=True,  on_epoch=True) # on_step=False, on_epoch=True
            # self.log(f"{stage}_acc", acc, prog_bar=True,  on_epoch=True) # on_step=False, on_epoch=True
            metrics = {f'{stage}_loss': loss, f"{stage}_acc": acc}
            # self.log_dict(self.metric.compute(predictions=preds, references=y), prog_bar=True)
            self.log_dict(metrics, prog_bar=True)
            metrics_table={
                f'{stage}_precision': precision(preds, y,average='macro',num_classes=2),
                f'{stage}_recall' : recall(preds, y,average='macro',num_classes=2),
                f'{stage}_f1_score' : f1(preds, y,average='macro',num_classes=2),
                #'avg_precision' : average_precision(preds, y),
                f'{stage}_mcc':matthews_corrcoef(preds, y,num_classes=2),
                
                }
            self.log_dict(metrics_table, prog_bar=True)
            y_np = y.cpu().detach().numpy()
            preds_np = preds.cpu().detach().numpy()
            metrics_table_sk={
                # 'precision sk': precision_score(y_np,preds_np,average='macro'),
                # 'recall sk' : recall_score(y_np,preds_np,average='macro'),
                f'{stage}_f1_score sk' : f1_score(y_np,preds_np),
                #'avg_precision' : average_precision(preds, y),
                f'{stage}_mcc sk':mcc(y_np,preds_np),
                #'classification_report':classification_report(y_np,preds_np)
                
                }

            self.log_dict(metrics_table_sk, prog_bar=True)
            cs_report = classification_report(y_np,preds_np,output_dict=True)
            print(cs_report)
            self.log_dict(cs_report, prog_bar=False)#
            #self.log('confusion_matrix' , confusion_matrix(preds, y,num_classes=2), prog_bar=False)


        return metrics

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        metrics = self.evaluate(batch, "val")
        return metrics['val_loss']
    
    
        

    def test_step(self, batch, batch_idx):
        print('Testing step')
        metrics = self.evaluate(batch, "test")
        return metrics

    def setup(self, stage=None) -> None:
        if stage != "fit":
            return
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        train_loader = self.train_dataloader()

        # Calculate total steps
        tb_size = self.hparams.train_batch_size * max(1, self.trainer.gpus)
        ab_size = float(self.trainer.accumulate_grad_batches) #* float(self.trainer.max_epochs) # TODO
        self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

    # def configure_optimizers(self):
    #     optimizer = torch.optim.SGD(
    #         self.parameters(),
    #         lr=self.hparams.learning_rate,
    #         momentum=0.9,
    #         weight_decay=5e-4,
    #     )
    #     #steps_per_epoch = 45000 // BATCH_SIZE
    #     print('----------self.total_steps : ',int(self.total_steps))
    #     scheduler_dict = {
    #         "scheduler": OneCycleLR(
    #             optimizer,
    #             0.1,
    #             epochs=self.trainer.max_epochs,
    #             steps_per_epoch=116 #int(self.total_steps), 
    #         ),
    #         "interval": "step",
    #     }
    #     return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon) # .hparams.

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

import argparse

def argparser():
    model_names = ['gpt2', 'bert-base-uncased', 'bert-base-cased','bert-large-cased','albert-large-v2',
    'distilbert-base-uncased', 'distilbert-base-uncased-finetuned-sst-2-english', 
    'roberta-large', 'roberta-base', 'roberta-large', 'albert-base-v2', 'distilbert-base-cased']
    tasks = ["cola", "mrpc"] 
    parser = argparse.ArgumentParser(description='NLP Project 1')
    parser.add_argument('--data', metavar='DIR',default='./data',
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='albert-base-v2',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: albert-base-v2)')
    parser.add_argument('--task', default='cola', type=str, choices=tasks,
                        help='task : ' +
                             ' | '.join(tasks) +
                             ' (default: cola)')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', '--learning_rate', default=5e-5, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        metavar='N',
                        help='mini-batch size (default: 16)')

    return parser.parse_args()

def prediction_data_loader():
    pass

def main():
    args = argparser()
    params = vars(args)
    print(params)
    AVAIL_GPUS = min(1, torch.cuda.device_count())
    NUM_WORKERS = 0 # int(os.cpu_count() / 2)
    TOKENIZERS_PARALLELISM = True 
    print(f"AVAIL_GPUS: {AVAIL_GPUS}")

    seed_everything(42)

    # Train
    dm = GLUEDataModule(
        model_name_or_path = params['arch'], 
        task_name = params['task'],
        num_workers = NUM_WORKERS,
        train_batch_size = params['batch_size'],
        eval_batch_size = params['batch_size']
        )
    dm.setup("fit")
    model = GLUETransformer(
        model_name_or_path = params['arch'],
        num_labels = dm.num_labels,
        eval_splits = dm.eval_splits,
        task_name = dm.task_name,
        learning_rate = params['lr'],
        train_batch_size = params['batch_size'],
        eval_batch_size = params['batch_size']
    )

    model_version = f"{args.task}-{args.arch}--e-{args.epochs}--lr-{model.hparams.learning_rate}--batch-{model.hparams.train_batch_size}"
    trainer = Trainer(
            max_epochs=args.epochs, 
            gpus=AVAIL_GPUS,
            logger=WandbLogger(project='IFT6285-NLP-Project1-runs-1',save_dir="lightning_logs/",name=model_version,log_model=False),#save_dir="lightning_logs/"
            #logger=TensorBoardLogger("lightning_logs/", name="cola",default_hp_metric=False),
            # callbacks=[LearningRateMonitor(logging_interval="step")]
        )
    # trainer.fit(model, train_dataloader=dm.train_dataloader(),val_dataloaders=dm.val_dataloader())
    # dm.setup("test")
    model = GLUETransformer.load_from_checkpoint('Models/MRPC/albert.ckpt')

    trainer.test(model, test_dataloaders=dm.test_dataloader())
    # trainer.test(model, test_dataloaders=dm.test_dataloader())
    # model = GLUETransformer.load_from_checkpoint('Models/MRPC/albert.ckpt')
    # model.freeze()
    # trainer.predict(model,dataloaders=dm.test_dataloader())
    # setup your data loader
    # test_dataloader = DataLoader(...)

    # # test (pass in the loader)
    # trainer.test(dataloaders=test_dataloader)




if __name__ == '__main__':
    main()