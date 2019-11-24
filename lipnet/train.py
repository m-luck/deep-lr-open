from torch import nn, optim
from torch.optim.optimizer import Optimizer
from torch import device
from lipnet.dataset import GridDataset
from lipnet.model import LipNet


def run(base_dir: str, use_overlapped: bool, batch_size: int, num_workers: int, target_device: device):
    train_dataset = GridDataset(base_dir, is_training=True, is_overlapped=use_overlapped)
    test_dataset = GridDataset(base_dir, is_training=False, is_overlapped=use_overlapped)

    loss_fn = nn.CTCLoss()

    model = LipNet.load()
    model.to(target_device)

    optimizer = optim.Adam(model.parameters(),
                           lr=2e-5,
                           weight_decay=0.,
                           amsgrad=True)

    train(model, train_dataset, optimizer, loss_fn, batch_size, num_workers, target_device)


def train(model: LipNet, train_dataset: GridDataset, optimizer: Optimizer, loss_fn: nn.CTCLoss, batch_size: int,
          num_workers: int, target_device: device):
    loader = train_dataset.get_data_loader(batch_size, num_workers, shuffle=True)

    train_cer = []
    for epoch in range(10):
        for (i, record) in enumerate(loader):
            model.train()
            images_tensor = record['images_tensor'].to(target_device)
            word_tensor = record['word_tensor'].to(target_device)
            images_length = record['images_length'].to(target_device)
            word_length = record['word_length'].to(target_device)

            optimizer.zero_grad()
            y = model(images_tensor)
            loss = loss_fn(y.transpose(0, 1).log_softmax(-1), word_tensor, images_length.view(-1), word_length.view(-1))
            loss.backward()

            optimizer.step()

            tot_iter = i + epoch * len(loader)

            pred_text = ctc_decode(y)
            actual_text = record["word_str"]

            if i % 100 == 0:
                for a, p in zip(actual_text, pred_text):
                    print("truth, pred: {}, {}".format(a, p))

            train_cer.extend(GridDataset.cer(pred_text, actual_text))


def ctc_decode(y):
    y = y.argmax(-1)
    return [GridDataset.convert_ctc_array_to_text(y[_]) for _ in range(y.size(0))]
