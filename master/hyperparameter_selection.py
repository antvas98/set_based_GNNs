import torch
import itertools
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from GNN_architectures import GCNconv_one_aggregator_Net, GCNconv_two_aggregators_Net, GINconv_one_aggregator_Net, GINconv_two_aggregators_Net, gcn_tuple_Net, gin_tuple2_Net, gin_tuple3_Net
def hyperparameter_selection_within_fold(data_train,
        data_test,
        combinations={'hidden_units':[16,32,64],'lr':[0.001], 'weight_decay':[0.00007]}, 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        batch_size=64,
        epochs = 50,
        aggregators='gcn 1'):
  # DataLoader for transformed dataset
  input_channels = data_train[0].x.shape[1]
  combinations_list = list(itertools.product(*combinations.values()))


  train_dataset, validation_dataset = train_test_split(data_train, test_size=0.1, random_state=42)
  train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
  val_loader = DataLoader(validation_dataset, batch_size)
  test_loader = DataLoader(data_test, batch_size)


  # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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

  for combination in combinations_list:
    # Create a dictionary with the current combination of values
    current_combination = dict(zip(combinations.keys(), combination))
    # print(current_combination)

    if aggregators=='gcn 1':
      model = GCNconv_one_aggregator_Net(input_channels = input_channels, hidden_units = current_combination['hidden_units']).to(device)
    if aggregators=='gcn 2':
      model = GCNconv_two_aggregators_Net(input_channels = input_channels, hidden_units = current_combination['hidden_units']).to(device)
    if aggregators=="gin 1":
      model = GINconv_one_aggregator_Net(input_channels = input_channels, hidden_units = current_combination['hidden_units']).to(device)
    if aggregators=='gin 2':
       model = GINconv_two_aggregators_Net(input_channels = input_channels, hidden_units = current_combination['hidden_units']).to(device)
    if aggregators=='gcn tuple2':
      model = gcn_tuple_Net(input_channels = input_channels, k=2, hidden_units = current_combination['hidden_units']).to(device)
    if aggregators=='gcn tuple3':
      model = gcn_tuple_Net(input_channels = input_channels, k=3, hidden_units = current_combination['hidden_units']).to(device)
    if aggregators=='gin tuple2':
      model = gin_tuple2_Net(input_channels = input_channels, k=2, hidden_units = current_combination['hidden_units']).to(device)
    if aggregators=='gin tuple3':
      model = gin_tuple3_Net(input_channels = input_channels, k=3, hidden_units = current_combination['hidden_units']).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=current_combination['lr'], weight_decay=current_combination['weight_decay'])

    best_acc_val = 0
    for epoch in range(1, epochs + 1):
        loss = train(model,optimizer)
        train_acc = test(model,train_loader)
        acc_val = test(model,val_loader)
        # log(Epoch=epoch, Loss=loss, Train=train_acc, Test=test_acc)
        if acc_val > best_acc_val:
          best_acc_val = acc_val
          best_combination = current_combination
          
  #Apply the best configuration on the test set
  if aggregators=='gcn 1':
    model = GCNconv_one_aggregator_Net(input_channels = input_channels, hidden_units = best_combination['hidden_units']).to(device)
  if aggregators=='gcn 2':
    model = GCNconv_two_aggregators_Net(input_channels = input_channels, hidden_units = best_combination['hidden_units']).to(device)
  if aggregators=="gin 1":
    model = GINconv_one_aggregator_Net(input_channels = input_channels, hidden_units = current_combination['hidden_units']).to(device)
  if aggregators=='gin 2':
      model = GINconv_two_aggregators_Net(input_channels = input_channels, hidden_units = current_combination['hidden_units']).to(device)
  if aggregators=='gcn tuple2':
      model = gcn_tuple_Net(input_channels = input_channels, k=2, hidden_units = current_combination['hidden_units']).to(device)
  if aggregators=='gcn tuple3':
      model = gcn_tuple_Net(input_channels = input_channels, k=3, hidden_units = current_combination['hidden_units']).to(device)
  if aggregators=='gin tuple2':
      model = gin_tuple2_Net(input_channels = input_channels, k=2, hidden_units = current_combination['hidden_units']).to(device)
  if aggregators=='gin tuple3':
      model = gin_tuple3_Net(input_channels = input_channels, k=3, hidden_units = current_combination['hidden_units']).to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=best_combination['lr'], weight_decay=best_combination['weight_decay'])
  best = 0
  for epoch in range(1, epochs + 1):
    loss = train(model,optimizer)
    acc_test = test(model,test_loader)
    if acc_test > best:
      best = acc_test
  return best