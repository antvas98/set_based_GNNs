import torch
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MLP, GINConv, global_add_pool


# does not discriminate between local and global neighbor (one aggregator)
class GCNconv_one_aggregator_Net(torch.nn.Module):
    def __init__(self, input_channels, hidden_units = 32):
        super(GCNconv_one_aggregator_Net, self).__init__()
        dim = hidden_units
        self.conv_1 = GCNConv(input_channels, dim) 
        self.mlp_1 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv_2 = GCNConv(dim, dim)
        self.mlp_2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.lin = Linear(dim, 1)

    def forward(self,data):
        x, edge_index, batch = data.x, data.edge_index.int(), data.batch # the train_loader convert int: 1 to float: 1.0000e+00 in the edge_index and produces an error

        # index = data.index

        x = F.relu(self.conv_1(x, edge_index))
        x = self.mlp_1(x)
        x = F.relu(self.conv_2(x, edge_index))
        x = self.mlp_2(x)
            
        x = global_add_pool(x, batch)
        
        x = self.lin(x)
        x = x.squeeze(dim=-1)

        return x #F.log_softmax(x)
    
# discriminates between local and global neighbors (2 different aggregators)
class GCNconv_two_aggregators_Net(torch.nn.Module):
  def __init__(self, input_channels, hidden_units = 32):
      super(GCNconv_two_aggregators_Net, self).__init__()
      dim = hidden_units
      self.conv_1_1 = GCNConv(input_channels, dim)
      self.conv_1_2 = GCNConv(input_channels, dim)

      self.mlp_1 = Sequential(Linear(2 * dim, dim), ReLU(), Linear(dim, dim))

      self.conv_2_1 = GCNConv(dim, dim)
      self.conv_2_2 = GCNConv(dim, dim)

      self.mlp_2 = Sequential(Linear(2 * dim, dim), ReLU(), Linear(dim, dim))

      self.lin = Linear(dim, 1)

  def forward(self,data):
    x, edge_index_local, edge_index_global, batch = data.x, data.edge_index_local.int(), data.edge_index_global.int(), data.batch


    x_1 = F.relu(self.conv_1_1(x, edge_index_local))
    x_2 = F.relu(self.conv_1_2(x, edge_index_global))
    x = self.mlp_1(torch.cat([x_1, x_2], dim=-1))

    x_1 = F.relu(self.conv_2_1(x, edge_index_local))
    x_2 = F.relu(self.conv_2_2(x, edge_index_global))
    x = self.mlp_2(torch.cat([x_1, x_2], dim=-1))

    x = global_add_pool(x, batch)
    
    x = self.lin(x)
    x = x.squeeze(dim=-1)
    return x 

class GINconv_one_aggregator_Net(torch.nn.Module):
    def __init__(self, input_channels, hidden_units=32, out_channels=1, num_layers=2):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([input_channels, hidden_units, hidden_units])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            input_channels = hidden_units
        self.mlp = MLP([hidden_units, hidden_units, out_channels],
                       norm=None, dropout=0.5)

    def forward(self, data): #I changed the arguements instead of x, edge_index, batch 
        x=data.x
        edge_index=data.edge_index
        batch=data.batch
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_add_pool(x, batch)
        return self.mlp(x).squeeze(dim=-1)
    
class GINconv_two_aggregators_Net(torch.nn.Module):
  def __init__(self, input_channels, hidden_units = 32):
      super(GINconv_two_aggregators_Net, self).__init__()
      dim = hidden_units
      self.conv_1_1 = GINConv(nn=MLP([input_channels, hidden_units, hidden_units]), train_eps=False) #GCNConv(input_channels, dim)
      self.conv_1_2 = GINConv(nn=MLP([input_channels, hidden_units, hidden_units]), train_eps=False) #GCNConv(input_channels, dim)

      self.mlp_1 = Sequential(Linear(2 * dim, dim), ReLU(), Linear(dim, dim))

      self.conv_2_1 = GINConv(nn=MLP([hidden_units, hidden_units, hidden_units]), train_eps=False) #GCNConv(dim, dim)
      self.conv_2_2 = GINConv(nn=MLP([hidden_units, hidden_units, hidden_units]), train_eps=False) #GCNConv(dim, dim)

      self.mlp_2 = Sequential(Linear(2 * dim, dim), ReLU(), Linear(dim, dim))

      self.lin = Linear(dim, 1)

  def forward(self,data):
    x, edge_index_local, edge_index_global, batch = data.x, data.edge_index_local, data.edge_index_global, data.batch


    x_1 = self.conv_1_1(x, edge_index_local)
    x_2 = self.conv_1_2(x, edge_index_global)
    x = self.mlp_1(torch.cat([x_1, x_2], dim=-1))

    x_1 = F.relu(self.conv_2_1(x, edge_index_local))
    x_2 = F.relu(self.conv_2_2(x, edge_index_global))
    x = self.mlp_2(torch.cat([x_1, x_2], dim=-1))

    x = global_add_pool(x, batch)
    
    x = self.lin(x)
    x = x.squeeze(dim=-1)
    return x 

class gcn_tuple_Net(torch.nn.Module):
    def __init__(self, input_channels, k=2, hidden_units = 32):
        super(gcn_tuple_Net, self).__init__()
        input_dim = input_channels 
        dim = hidden_units
        self.k = k
        self.convs1 = torch.nn.ModuleList([GCNConv(input_dim, dim) for _ in range(self.k)])
        self.convs2 = torch.nn.ModuleList([GCNConv(dim, dim) for _ in range(self.k)])

        self.mlp_1 = Sequential(Linear(self.k * dim, dim), ReLU(), Linear(dim, dim))
        self.mlp_2 = Sequential(Linear(self.k * dim, dim), ReLU(), Linear(dim, dim))
        self.lin = Linear(dim, 1)

    def forward(self, data):
      x, batch = data.x, data.batch
      edge_indices = [getattr(data, f'edge_index_{i}') for i in range(self.k)]
      indices = [getattr(data,f'index_{i}') for i in range(self.k)]

      for i in range(self.k):
          x_i = F.relu(self.convs1[i](x, edge_indices[i]))
          setattr(self, f'x_{i}', x_i)

      x = self.mlp_1(torch.cat([getattr(self, f'x_{i}') for i in range(self.k)], dim=-1))

      for i in range(self.k):
          x_i = F.relu(self.convs2[i](x, edge_indices[i]))
          setattr(self, f'x_{i}', x_i)

      x = self.mlp_2(torch.cat([getattr(self, f'x_{i}') for i in range(self.k)], dim=-1))
      x = global_add_pool(x, batch)
      x = self.lin(x)
      x = x.squeeze(dim=-1)

      return x

class gin_tuple2_Net(torch.nn.Module):
  def __init__(self, input_channels, k=2, hidden_units = 32):
      super(gin_tuple2_Net, self).__init__()
      dim = hidden_units
      self.conv_1_1 = GINConv(nn=MLP([input_channels, hidden_units, hidden_units]), train_eps=False) #GCNConv(input_channels, dim)
      self.conv_1_2 = GINConv(nn=MLP([input_channels, hidden_units, hidden_units]), train_eps=False) #GCNConv(input_channels, dim)

      self.mlp_1 = Sequential(Linear(2 * dim, dim), ReLU(), Linear(dim, dim))

      self.conv_2_1 = GINConv(nn=MLP([hidden_units, hidden_units, hidden_units]), train_eps=False) #GCNConv(dim, dim)
      self.conv_2_2 = GINConv(nn=MLP([hidden_units, hidden_units, hidden_units]), train_eps=False) #GCNConv(dim, dim)

      self.mlp_2 = Sequential(Linear(2 * dim, dim), ReLU(), Linear(dim, dim))

      self.lin = Linear(dim, 1)

  def forward(self,data):
    x, edge_index_0, edge_index_1, batch = data.x, data.edge_index_0, data.edge_index_1, data.batch


    x_0 = self.conv_1_1(x, edge_index_0)
    x_1 = self.conv_1_2(x, edge_index_1)
    x = self.mlp_1(torch.cat([x_0, x_1], dim=-1))

    x_0 = F.relu(self.conv_2_1(x, edge_index_0))
    x_1 = F.relu(self.conv_2_2(x, edge_index_1))
    x = self.mlp_2(torch.cat([x_0, x_1], dim=-1))

    x = global_add_pool(x, batch)
    
    x = self.lin(x)
    x = x.squeeze(dim=-1)
    return x
  
class gin_tuple3_Net(torch.nn.Module):
  def __init__(self, input_channels, k=3, hidden_units = 32):
      super(gin_tuple3_Net, self).__init__()
      dim = hidden_units
      self.conv_1_1 = GINConv(nn=MLP([input_channels, hidden_units, hidden_units]), train_eps=False) #GCNConv(input_channels, dim)
      self.conv_1_2 = GINConv(nn=MLP([input_channels, hidden_units, hidden_units]), train_eps=False) #GCNConv(input_channels, dim)
      self.conv_1_3 = GINConv(nn=MLP([input_channels, hidden_units, hidden_units]), train_eps=False) #GCNConv(input_channels, dim)

      self.mlp_1 = Sequential(Linear(3 * dim, dim), ReLU(), Linear(dim, dim))

      self.conv_2_1 = GINConv(nn=MLP([hidden_units, hidden_units, hidden_units]), train_eps=False) #GCNConv(dim, dim)
      self.conv_2_2 = GINConv(nn=MLP([hidden_units, hidden_units, hidden_units]), train_eps=False) #GCNConv(dim, dim)
      self.conv_2_3 = GINConv(nn=MLP([hidden_units, hidden_units, hidden_units]), train_eps=False) #GCNConv(dim, dim)

      self.mlp_2 = Sequential(Linear(3 * dim, dim), ReLU(), Linear(dim, dim))

      self.lin = Linear(dim, 1)

  def forward(self,data):
    x, edge_index_0, edge_index_1, edge_index_2, batch = data.x, data.edge_index_0, data.edge_index_1, data.edge_index_2, data.batch


    x_0 = self.conv_1_1(x, edge_index_0)
    x_1 = self.conv_1_2(x, edge_index_1)
    x_2 = self.conv_1_3(x, edge_index_2)
    x = self.mlp_1(torch.cat([x_0, x_1, x_2], dim=-1))

    x_0 = F.relu(self.conv_2_1(x, edge_index_0))
    x_1 = F.relu(self.conv_2_2(x, edge_index_1))
    x_2 = F.relu(self.conv_2_3(x, edge_index_2))
    x = self.mlp_2(torch.cat([x_0, x_1, x_2], dim=-1))

    x = global_add_pool(x, batch)
    
    x = self.lin(x)
    x = x.squeeze(dim=-1)
    return x     