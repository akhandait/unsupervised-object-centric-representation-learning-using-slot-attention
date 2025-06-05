import os
import argparse
from model import *
from tqdm import tqdm
import time
import datetime
import torch.optim as optim
import torch
from dataset import *
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument
parser.add_argument('--models_dir', default='./trained_models/', type=str, help='where to save models' )
parser.add_argument('--losses_dir', default='./losses/', type=str, help='where to save losses')
parser.add_argument('--save_model_id', default=1, type=int, help='model id to save')
parser.add_argument('--load_model_id', default=-1, type=int, help='model id to load')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_slots', default=30, type=int, help='Number of slots in Slot Attention.')
parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations.')
parser.add_argument('--hid_dim', default=32, type=int, help='hidden dimension size')
parser.add_argument('--learning_rate', default=0.0004, type=float)
parser.add_argument('--warmup_steps', default=10000, type=int, help='Number of warmup steps for the learning rate.')
parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for the learning rate decay.')
parser.add_argument('--decay_steps', default=100000, type=int, help='Number of steps for the learning rate decay.')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')
parser.add_argument('--num_epochs', default=100, type=int, help='number of workers for loading data')
parser.add_argument('--data_path', default='../Dataset/', type=str, help='path to the data folder')
parser.add_argument('--temperature', default=None, type=float, help='temperature for the softmax in the attention module')
parser.add_argument('--l_ent', default=0.01, type=float, help='lambda for the entropy loss, tune 0.01 - 0.10')
parser.add_argument('--l_rep', default=0.05, type=float, help='lambda for the repulsion loss, tune 0.05 - 0.20')
parser.add_argument('--base_sigma', default=0.2, type=float, help='base sigma for the slots')

# python main.py --product_id 1001028

def main():
    opt = parser.parse_args()

    # ------------------------------- CONFIG --------------------------------------
    PATCH_FOLDERS = [opt.data_path + "01_PATCHES", opt.data_path + "../Dataset/02_PATCHES"]   # the data folders
    SPLIT = (0.70, 0.15, 0.15)                     # train / val / test ratio
    SEED = 42                                      # reproducible shuffling
    # -----------------------------------------------------------------------------

    resolution = (160, 240)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device: ' + str(device))

    print("Building complete dataset …")
    dataset = PatchDataset(PATCH_FOLDERS)

    print(f"Total patches: {len(dataset):,}")

    print("Splitting (train / val / test) …")
    train_ds, val_ds, test_ds = split_dataset(dataset, SPLIT, SEED)

    model = SlotAttentionAutoEncoder(resolution, opt.num_slots, opt.num_iterations, opt.hid_dim,\
                                     opt.base_sigma, opt.temperature).to(device)
    if opt.load_model_id != -1:
        model.load_state_dict(torch.load(opt.models_dir + 'model' + str(opt.load_model_id) + '.ckpt')['model_state_dict'])

    criterion = nn.MSELoss()
    λ_ent = opt.l_ent * 1e-3
    λ_rep = opt.l_rep * 1e-4

    params = [{'params': model.parameters()}]

    train_dataloader = DataLoader(
        train_ds,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        collate_fn=collate_with_meta)
    
    val_dataloader = DataLoader(
        val_ds,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        collate_fn=collate_with_meta)

    optimizer = optim.Adam(params, lr=opt.learning_rate)
    
    with torch.no_grad():
        model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_ent_loss = 0
        val_rep_loss = 0
        for sample, meta in tqdm(val_dataloader):
            image = sample.to(device)
            recon_combined, recons, masks, slots, loss_entropy, loss_repulsion = model(image)
            recon_loss = criterion(recon_combined, image)
            loss = recon_loss + λ_ent * loss_entropy + λ_rep * loss_repulsion

            val_loss += loss.item()
            val_recon_loss += recon_loss.item()
            val_ent_loss += loss_entropy.item()
            val_rep_loss += loss_repulsion.item()

        val_loss /= len(val_dataloader)
        val_recon_loss /= len(val_dataloader)
        val_ent_loss /= len(val_dataloader)
        val_rep_loss /= len(val_dataloader)
    
    print ("Initial Validation Losses: rec={}, ent={}, rep={}".format(val_recon_loss, val_ent_loss, val_rep_loss))
    
    train_recon_losses, train_ent_losses, train_rep_losses = [], [], []
    val_recon_losses, val_ent_losses, val_rep_losses = [], [], []

    start = time.time()
    i = 0
    for epoch in range(opt.num_epochs):
        model.train()

        total_loss = 0
        total_recon_loss = 0
        total_ent_loss = 0
        total_rep_loss = 0
        
        for sample, meta in tqdm(train_dataloader):
            i += 1

            if i < opt.warmup_steps:
                learning_rate = opt.learning_rate * (i / opt.warmup_steps)
            else:
                learning_rate = opt.learning_rate

            learning_rate = learning_rate * (opt.decay_rate ** (
                i / opt.decay_steps))

            optimizer.param_groups[0]['lr'] = learning_rate
            
            image = sample.to(device)
            recon_combined, recons, masks, slots, loss_entropy, loss_repulsion = model(image)
            recon_loss = criterion(recon_combined, image)
        
            loss = recon_loss + λ_ent * loss_entropy + λ_rep * loss_repulsion
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_ent_loss += loss_entropy.item()
            total_rep_loss += loss_repulsion.item()

            # print('Entropy Loss: ' + str(loss_entropy.item()))
            # print('Repulsion Loss: ' + str(loss_repulsion.item()))
  
            del recons, masks, slots

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss /= len(train_dataloader)
        total_recon_loss /= len(train_dataloader)
        total_ent_loss /= len(train_dataloader)
        total_rep_loss /= len(train_dataloader)
        
        train_recon_losses.append(total_recon_loss)
        train_ent_losses.append(total_ent_loss)
        train_rep_losses.append(total_rep_loss)

        with torch.no_grad():
            model.eval()
            val_loss = 0
            val_recon_loss = 0
            val_ent_loss = 0
            val_rep_loss = 0
            for sample, meta in tqdm(val_dataloader):
                image = sample.to(device)
                recon_combined, recons, masks, slots, loss_entropy, loss_repulsion = model(image)
                recon_loss = criterion(recon_combined, image)
                loss = recon_loss + λ_ent * loss_entropy + λ_rep * loss_repulsion

                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_ent_loss += loss_entropy.item()
                val_rep_loss += loss_repulsion.item()

            val_loss /= len(val_dataloader)
            val_recon_loss /= len(val_dataloader)
            val_ent_loss /= len(val_dataloader)
            val_rep_loss /= len(val_dataloader)
        
        print ("Epoch: {}, Training Losses: rec={}, Time: {}".format(epoch, total_recon_loss,
            datetime.timedelta(seconds=time.time() - start)))
        print ("Epoch: {}, Validation Losses: rec={}, ent={}, rep={}".format(epoch, val_recon_loss, val_ent_loss, val_rep_loss))
        
        val_recon_losses.append(val_recon_loss)
        val_ent_losses.append(val_ent_loss)
        val_rep_losses.append(val_rep_loss)
        
        with open(opt.losses_dir + 'losses_model' + str(opt.save_model_id) + '.pkl', 'wb') as f:
            pickle.dump([train_recon_losses, train_ent_losses, train_rep_losses, val_recon_losses, val_ent_losses, val_rep_losses], f)

        if epoch % 10 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                }, opt.models_dir + 'model' + str(opt.save_model_id) + '.ckpt')
            print("Model saved.")

if __name__ == "__main__":
    main()
