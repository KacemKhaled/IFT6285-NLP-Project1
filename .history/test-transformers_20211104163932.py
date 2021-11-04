from datetime import datetime
from typing import Optional

import os
import datasets
import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.functional import accuracy #, precision, recall, f1, average_precision, confusion_matrix
import torch.nn.functional as F
from pytorch_lightning.callbacks import LearningRateMonitor



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
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
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
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size,num_workers=self.num_workers)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size,num_workers=self.num_workers) for x in self.eval_splits]

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




class GLUETransformer(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        task_name: str,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)#, config=self.config)
        self.metric = datasets.load_metric(
            "glue", self.hparams.task_name, experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        # x, y = batch
        y = batch['labels']
        outputs = self(**batch)
        loss, logits =  outputs[:2]
        if self.hparams.num_labels >= 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()
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
        # x, y = batch
        y = batch['labels']
        outputs = self(**batch)
        loss, logits =  outputs[:2]
        if self.hparams.num_labels >= 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()
        # logits = self(x)
        # loss = F.nll_loss(logits, y)
        # preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            stage = ''
            self.log(f"{stage}_loss", loss, prog_bar=True,  on_epoch=True) # on_step=False, on_epoch=True
            self.log(f"{stage}_acc", acc, prog_bar=True,  on_epoch=True) # on_step=False, on_epoch=True
            metrics = {f'{stage}_loss': loss, f"{stage}_acc": acc}
            self.log_dict(self.metric.compute(predictions=preds, references=y), prog_bar=True)
            # if stage=='test':
            #     metrics.update({
            #         'precision': precision(preds, y),
            #         'recall' : recall(preds, y),
            #         'f1_score' : f1(preds, y),
            #         #'avg_precision' : average_precision(preds, y),
            #         #'confusion_matrix' : confusion_matrix(preds, y,num_classes=2),
            #         })
            # self.log_dict(metrics, prog_bar=True)

        return metrics

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        metrics = self.evaluate(batch, "val")
        return metrics
        

    # def test_step(self, batch, batch_idx):
    #     metrics = self.evaluate(batch, "test")
    #     return metrics

    def setup(self, stage=None) -> None:
        if stage != "fit":
            return
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        train_loader = self.train_dataloader()

        # Calculate total steps
        tb_size = self.hparams.train_batch_size * max(1, self.trainer.gpus)
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

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
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

import argparse

def argparser():
    model_names = ['gpt2', 'bert-base-uncased', 'bert-base-cased', 'distilbert-base-uncased-finetuned-sst-2-english', 'roberta-large', 'roberta-base', 'roberta-large', 'albert-base-v2', 'distilbert-base-cased']
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
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256)')

    return parser.parse_args()



def main():
    args = argparser()
    AVAIL_GPUS = min(1, torch.cuda.device_count())
    NUM_WORKERS = 0 # int(os.cpu_count() / 2)
    TOKENIZERS_PARALLELISM = True 
    print(f"AVAIL_GPUS: {AVAIL_GPUS}")

    seed_everything(42)

    # Train
    dm = GLUEDataModule(model_name_or_path=args.arch, task_name=args.task,num_workers=NUM_WORKERS)
    dm.setup("fit")
    model = GLUETransformer(
        model_name_or_path=args.arch,
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        task_name=dm.task_name,
    )

    model_version = f"{args.task}--a-{args.arch}--e-{args.epochs}"
    trainer = Trainer(
        max_epochs=args.epochs, 
        gpus=AVAIL_GPUS,
        logger=WandbLogger(save_dir="lightning_logs/",name=model_version,log_model=True),
        #logger=TensorBoardLogger("lightning_logs/", name="cola",default_hp_metric=False),
        callbacks=[LearningRateMonitor(logging_interval="step")]

        )
    trainer.fit(model, dm)
    # trainer.test(model, dm)




if __name__ == '__main__':
    main()