import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from loguru import logger
from tqdm import tqdm
from transformers import set_seed

from utils.args import get_parser
from utils.optim import get_linear_schedule_with_warmup
from model.models import BertNER
from reader.nerReader import NERSet

args = get_parser()


def main():
    writer = SummaryWriter(args.log_dir)
    if args.no_cuda or not torch.cuda.is_available():
        DEVICE = torch.device('cpu')
        logger.info('use cpu!')
    else:
        DEVICE = torch.device('cuda', 0)
        logger.info('use gpu!')

    set_seed(args.random_seed)

    trainset = NERSet(args, 'train', True)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers,
                             shuffle=True, collate_fn=NERSet.collate)

    model = BertNER(args, DEVICE)
    model = model.to(DEVICE)
    optimizer = AdamW([{'params': model.encoder.parameters()},
                       {'params': model.emission_ffn.parameters()},
                       {'params': model.crf.parameters(), "lr": 1e-3}], lr=args.learning_rate)

    global_step = 0
    for epoch in range(args.max_epochs):
        with tqdm(total=len(trainloader)) as t:
            t.set_description(f'Epoch {epoch}')
            model.train()
            for model_inputs, sample_infos in trainloader:
                for k, v in model_inputs.items():
                    if isinstance(v, torch.Tensor):
                        model_inputs[k] = v.to(DEVICE)

                global_step += 1
                loss = model(model_inputs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                writer.add_scalar('crf-loss', loss, global_step=global_step)
                t.set_postfix(loss=loss)
                t.update(1)

if __name__ == '__main__':
    main()