import os

import sys

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

class DrugDataset(Dataset):

    def __init__(self, data_dir, transform=None):

        self.data_dir = data_dir

        self.transform = transform

        self.smiles = []

        self.targets = []

        for file in os.listdir(data_dir):

            with open(os.path.join(data_dir, file), 'r') as f:

                smiles, target = f.readline().split()

                self.smiles.append(smiles)

                self.targets.append(target)

    def __len__(self):

        return len(self.smiles)

    def __getitem__(self, idx):

        smiles = self.smiles[idx]

        target = self.targets[idx]

        if self.transform:

            smiles = self.transform(smiles)

        return smiles, target

class GCN(nn.Module):

    def __init__(self, in_features, hidden_features, out_features):

        super(GCN, self).__init__()

        self.conv1 = nn.Conv1d(in_features, hidden_features, kernel_size=1)

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(hidden_features, out_features, kernel_size=1)
def forward(self, x, edge_index):

        x = self.conv1(x)

        x = self.relu(x)

        x = self.conv2(x)

        return x

def train(model, device, train_loader, optimizer, epoch):

    model.train()

    for batch in train_loader:

        smiles, targets = batch

        smiles = smiles.to(device)

        targets = targets.to(device)

        optimizer.zero_grad()

        output = model(smiles, edge_index)

        loss = F.cross_entropy(output, targets)

        loss.backward()

        optimizer.step()

    print('Train Epoch: {} Loss: {:.4f}'.format(epoch, loss.item()))

def test(model, device, test_loader):

    model.eval()

    correct = 0

    total = 0

    for batch in test_loader:

        smiles, targets = batch

        smiles = smiles.to(device)

        targets = targets.to(device)

        output = model(smiles, edge_index)

        pred = output.argmax(dim=1)

        correct += (pred == targets).sum().item()

        total += targets.size(0)

    print('Test Accuracy: {:.4f}'.format(correct / total))

if __name__ == '__main__':

    data_dir = './data'
# Load the data

    train_dataset = DrugDataset(data_dir, transform=None)

    test_dataset = DrugDataset(data_dir, transform=None)

    # Create data loaders

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=32)

    # Create the model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GCN(784, 128, 10).to(device)

    # Create the optimizer

    optimizer = optim.Adam(model.parameters())

    # Train the model

    for epoch in range(10):

        train(model, device, train_loader, optimizer, epoch)

        test(model, device, test_loader)
# Save the model

torch.save(model.state_dict(), 'model.pt')

# Load the model

model = GCN(784, 128, 10).to(device)

model.load_state_dict(torch.load('model.pt'))

# Fine-tune the model

for epoch in range(10):

    train(model, device, train_loader, optimizer, epoch)

    test(model, device, test_loader)

# Evaluate the model

print('Test Accuracy: {:.4f}'.format(test(model, device, test_loader)))

# Plot the training and test loss curves

plt.plot(train_losses, label='Train Loss')

plt.plot(test_losses, label='Test Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend()

plt.show()

# Visualize the learned features

for i in range(10):

    features = model.features[i].cpu().detach().numpy()

    plt.figure()

    plt.imshow(features)

    plt.title('Feature {}'.format(i))

    plt.show()

# Use the model to predict the properties of new drug compounds

new_compounds = ['CC(=O)O', 'NC(=O)O']

smiles = [Chem.MolFromSmiles(compound) for compound in new_compounds]

edge_index = get_edge_index(smiles)

features = model.features(smiles, edge_index)

# Print the predicted properties of the new drug compounds

for i, compound in enumerate(new_compounds):

    print('Predicted properties of compound {}:'.format(compound))

    print('    Molecular weight: {}'.format(features[i, 0]))

    print('    LogP: {}'.format(features[i, 1]))

    print('    Polar surface area: {}'.format(features[i, 2]))

    print('    Number of rotatable bonds: {}'.format(features[i, 3]))
# Generate new drug compounds

new_compounds = []

for i in range(1000):

    new_compound = Chem.MolFromSmiles('CC(=O)O')

    new_compound.AddAtom(Chem.Atom('N', 0, 0, 0))

    new_compound.AddBond(new_compound.GetAtomWithIdx(0), new_compound.GetAtomWithIdx(1), Chem.BondType.SINGLE)

    new_compounds.append(new_compound)

# Predict the properties of the new drug compounds

smiles = [Chem.MolToSmiles(compound) for compound in new_compounds]

edge_index = get_edge_index(smiles)

features = model.features(smiles, edge_index)

# Print the predicted properties of the new drug compounds

for i, compound in enumerate(new_compounds):

    print('Predicted properties of compound {}:'.format(compound))

    print('    Molecular weight: {}'.format(features[i, 0]))

    print('    LogP: {}'.format(features[i, 1]))

    print('    Polar surface area: {}'.format(features[i, 2]))

    print('    Number of rotatable bonds: {}'.format(features[i, 3]))

# Save the predicted properties of the new drug compounds to a file

with open('predicted_properties.csv', 'w') as f:

    writer = csv.writer(f)

    writer.writerow(['Compound', 'Molecular weight', 'LogP', 'Polar surface area', 'Number of rotatable bonds'])

    for i, compound in enumerate(new_compounds):

        writer.writerow([compound, features[i, 0], features[i, 1], features[i, 2], features[i, 3]])

# Load the predicted properties of the new drug compounds from a file

with open('predicted_properties.csv', 'r') as f:

    reader = csv.reader(f)

    predicted_properties = []

    for row in reader:

        compound, molecular_weight, logp, polar_surface_area, number_of_rotatable_bonds = row

        predicted_properties.append((compound, molecular_weight, logp, polar_surface_area, number_of_rotatable_bonds))

# Find the best new drug compound

best_compound = None

best_score = -1

for compound, molecular_weight, logp, polar_surface_area, number_of_rotatable_bonds in predicted_properties:

    score = molecular_weight + logp + polar_surface_area + number_of_rotatable_bonds

    if score > best_score:

        best_compound = compound

        best_score = score
# Print the best new drug compound

print('The best new drug compound is {} with a score of {}'.format(best_compound, best_score))
# Visualize the best new drug compound

smiles = Chem.MolToSmiles(best_compound)

edge_index = get_edge_index(smiles)

features = model.features(smiles, edge_index)

plt.figure()

plt.imshow(features)

plt.title('Best new drug compound')

plt.show()

# Evaluate the best new drug compound

new_compound = Chem.MolFromSmiles(smiles)

# Calculate the molecular weight of the new compound

molecular_weight = new_compound.GetMolecularWeight()

# Calculate the logP of the new compound

logp = new_compound.GetLogP()

# Calculate the polar surface area of the new compound

polar_surface_area = new_compound.GetPolarSurfaceArea()

# Calculate the number of rotatable bonds in the new compound

number_of_rotatable_bonds = new_compound.GetNumRotatableBonds()

# Print the properties of the best new drug compound

print('The best new drug compound has the following properties:')

print('    Molecular weight: {}'.format(molecular_weight))

print('    LogP: {}'.format(logp))

print('    Polar surface area: {}'.format(polar_surface_area))

print('    Number of rotatable bonds: {}'.format(number_of_rotatable_bonds))
# Find the best new drug compound

best_compound = None

best_score = -1

for compound, molecular_weight, logp, polar_surface_area, number_of_rotatable_bonds in predicted_properties:

    score = molecular_weight + logp + polar_surface_area + number_of_rotatable_bonds

    if score > best_score:

        best_compound = compound

        best_score = score

# Print the best new drug compound

print('The best new drug compound is {} with a score of {}'.format(best_compound, best_score))

# End the program

input('Press Enter to end the program.')
