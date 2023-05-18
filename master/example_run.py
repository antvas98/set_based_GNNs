from GNN_architectures import two_aggregators_Net
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import torch
from torch_geometric.logging import init_wandb, log
import argparse


def STT(location,
        dataset_name='MUTAG',
        hidden_units = 32,
        lr = 0.001,
        weight_decay = 0.00007,
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        batch_size=64,
        epochs = 50,
        num_aggregators=1):
    # DataLoader for transformed dataset
    data = torch.load(f'{location}/transformed_data/{dataset_name}/multiset2_delta_dataset_{dataset_name}.pt')
    input_channels = data[0].x.shape[1]
    configuration={'hidden_units':[hidden_units],'lr':[lr], 'weight_decay':[weight_decay]}

    train_dataset, validation_dataset = train_test_split(data, test_size=0.1, random_state=42)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size)

    # train function 
    def train(model,optimizer):
        model.train()

        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.binary_cross_entropy_with_logits(out, data.y.float()) 
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * data.num_graphs
        return total_loss / len(train_loader.dataset)

    # test function
    @torch.no_grad()
    def test(model,loader):
        model.eval()

        total_correct = 0
        i=0
        for data in loader:
            i+=1
            data = data.to(device)
            z = torch.sigmoid(model(data))
            pred = torch.where(z>0.5,1,0)
            total_correct += int((pred == data.y).sum())
        return total_correct/len(loader.dataset)

    model = two_aggregators_Net(input_channels = input_channels, hidden_units = hidden_units).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_acc_val = 0
    for epoch in range(1, epochs + 1):
        loss = train(model,optimizer)
        train_acc = test(model,train_loader)
        acc_val = test(model,val_loader)
        log(Epoch=epoch, Loss=loss, Train=train_acc, Test=acc_val)
        if acc_val > best_acc_val:
            best_acc_val = acc_val          
  
    return f"best accuracy is {best_acc_val}!"


parser = argparse.ArgumentParser()
parser.add_argument('--location', type=str, default='DATA',
                    help='input the path where is the folder named transformed_data. The default value is a folder named DATA')
parser.add_argument('--dataset_name', type=str, default='MUTAG',
                    help='input the name of the dataset (e.g MUTAG,PTC_FM, PTC_MR)')
parser.add_argument('--hidden_units', type=int, default=32,
                    help='number of hidden units for each layer')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs')
args, unknown = parser.parse_known_args() 

if __name__ == "__main__":
    STT(location=args.location, dataset_name=args.dataset_name,hidden_units=args.hidden_units, epochs=args.epochs)