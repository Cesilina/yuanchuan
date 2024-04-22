import os
from random import shuffle

import torch
import logging
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from tqdm import tqdm
from config import root_path
from classify.data import SSTDataset

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)
train_file = os.path.join(root_path, 'data/processed/train.csv')
# test_file = os.path.join(root_path, 'data/test.csv')
dev_file = os.path.join(root_path, 'data/processed/dev.csv')

def train_one_epoch(model, lossfn, optimizer, dataset, batch_size=8):
    generator = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    train_loss, train_acc = 0.0, 0.0
    for batch, labels in tqdm(generator):
        batch, labels = batch.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(batch, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        err = lossfn(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += err.item()
        pred_labels = torch.argmax(logits, axis=1)
        train_acc += (pred_labels == labels).sum().item()
    train_loss /= len(dataset)
    train_acc /= len(dataset)
    return train_loss, train_acc


def evaluate_one_epoch(model, lossfn, optimizer, dataset, batch_size=6):
    generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    model.eval()
    loss, acc = 0.0, 0.0
    with torch.no_grad():
        for batch, labels in tqdm(generator):
            batch, labels = batch.to(device), labels.to(device)
            logits = model(batch)[0]
            error = lossfn(logits, labels)
            loss += error.item()
            pred_labels = torch.argmax(logits, axis=1)
            acc += (pred_labels == labels).sum().item()
    loss /= len(dataset)
    acc /= len(dataset)
    return loss, acc


def train():

    batch_size = 10

    vocab_file = os.path.join(root_path, 'lib/bert/vocab.txt')
    tokenizer = BertTokenizer.from_pretrained(vocab_file)

    trainset = SSTDataset(train_file, tokenizer=tokenizer, maxlen=300)
    devset = SSTDataset(dev_file, tokenizer=tokenizer, maxlen=300)
    # testset = SSTDataset(test_file, tokenizer=tokenizer, maxlen=300)
    config_path = os.path.join(root_path, 'lib/bert')
    model_path = os.path.join(root_path, 'lib/bert/pytorch_model.bin')
    config = BertConfig.from_pretrained(config_path)
    config.num_labels = 501
    model = BertForSequenceClassification.from_pretrained(model_path, config=config)

    model = model.to(device)
    lossfn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    for epoch in range(1, 30):
        train_loss, train_acc = train_one_epoch(
            model, lossfn, optimizer, trainset, batch_size=batch_size
        )
        print("epoch：{}".format(epoch))
        print(
            "train_loss={}, train_acc={}".format(train_loss, train_acc)
        )
        if (epoch + 1) / 10 == 0:
            val_loss, val_acc = evaluate_one_epoch(
                model, lossfn, optimizer, devset, batch_size=batch_size
            )
        # test_loss, test_acc = evaluate_one_epoch(
        #     model, lossfn, optimizer, testset, batch_size=batch_size
        # )
            print(
                "val_loss={}, val_acc={}".format(val_loss, val_acc)
            )
        
            save_path = os.path.join(root_path, 'result/bert/' + "{}.pickle".format(epoch))
            torch.save(model, save_path)

    logger.info("Done!")


def main():
    """Train BERT sentiment classifier."""
    train()


if __name__ == "__main__":
    main()
