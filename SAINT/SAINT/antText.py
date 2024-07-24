import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import argparse
import time
from pretrainmodel import SAINT


def calculate_metrics(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    PD = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
    PF = cm[0, 1] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    balance = 1 - np.sqrt((1 - PD) ** 2 + PF ** 2) / np.sqrt(2)
    FI = (cm[1, 1] + cm[0, 1]) / (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1]) if (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1]) > 0 else 0
    FIR = (PD - FI) / PD if PD > 0 else 0
    return PD, PF, balance, FIR


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--batchsize', default=64, type=int)
    parser.add_argument('--run_name', default='testrun', type=str)
    parser.add_argument('--set_seed', default=1, type=int)
    parser.add_argument('--dset_seed', default=5, type=int)
    parser.add_argument('--active_log', action='store_true')
    return parser.parse_args()


def setup_fold_data(train_index, val_index, X_train, y_train, scaler, smote, opt):
    X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]
    X_fold_train_normalized = scaler.fit_transform(X_fold_train)
    X_fold_val_normalized = scaler.transform(X_fold_val)
    X_fold_train_resampled, y_fold_train_resampled = smote.fit_resample(X_fold_train_normalized, y_fold_train)
    X_train_tensor = torch.tensor(X_fold_train_resampled).float()
    y_train_tensor = torch.tensor(y_fold_train_resampled).long()
    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True, num_workers=4)
    X_valid_tensor = torch.tensor(X_fold_val_normalized).float()
    y_valid_tensor = torch.tensor(y_fold_val.to_numpy()).long()
    val_ds = TensorDataset(X_valid_tensor, y_valid_tensor)
    valloader = DataLoader(val_ds, batch_size=opt.batchsize, shuffle=False, num_workers=4)
    return trainloader, valloader


def evaluate_model(model, dataloader, device):
    model.eval()
    y_preds = []
    y_gts = []
    with torch.no_grad():
        for data in dataloader:
            x_cont = data[0].to(device).float()
            y_gts.extend(data[1].tolist())
            x_cont_enc = torch.cat([model.simple_MLP[i](x_cont[:, i:i + 1]) for i in range(model.num_continuous)], dim=-1)
            x_combined = model.fc(x_cont_enc)
            y_outs = model.mlpfory(x_combined)
            y_probs = torch.nn.functional.softmax(y_outs, dim=1)
            y_preds.extend(torch.argmax(y_probs, dim=1).tolist())
    return torch.tensor(y_gts), torch.tensor(y_preds)


class AntColonyOptimizer:
    def __init__(self, objective_function, search_space, n_ants, n_best, n_iterations, decay, alpha=1, beta=1):
        self.objective_function = objective_function
        self.search_space = search_space
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha  # Influence of pheromone
        self.beta = beta  # Influence of heuristic information

        # Initialize pheromones for each hyperparameter in the search space
        self.pheromones = {key: np.ones(len(values)) for key, values in search_space.items()}

    def _select_param(self, pheromones, values):
        # Calculate probabilities for each value based on pheromone levels and heuristic information
        pheromone_levels = np.array(pheromones)
        probabilities = pheromone_levels ** self.alpha
        probabilities /= probabilities.sum()
        return np.random.choice(values, p=probabilities)

    def _generate_ant_params(self):
        ant_params = {}
        for param, values in self.search_space.items():
            selected_value = self._select_param(self.pheromones[param], values)
            ant_params[param] = selected_value
        return ant_params

    def optimize(self):
        best_params = None
        best_score = float('-inf')

        for iteration in range(self.n_iterations):
            all_params = []
            all_scores = []

            for _ in range(self.n_ants):
                params = self._generate_ant_params()
                score = self.objective_function(params)
                all_params.append(params)
                all_scores.append(score)

                if score > best_score:
                    best_score = score
                    best_params = params

            # Update pheromones
            sorted_indices = np.argsort(all_scores)[-self.n_best:]
            for param in self.search_space:
                self.pheromones[param] *= self.decay
                for i in sorted_indices:
                    value_index = self.search_space[param].index(all_params[i][param])
                    self.pheromones[param][value_index] += 1.0

        return best_params, best_score


def objective_function(params):
    print(f"Training with parameters: {params}")
    model = SAINT(
        num_continuous=len(X_train.columns),
        dim=params['dim'],
        depth=params['depth'],
        heads=8,  # Fixed
        attn_dropout=0.1,  # Fixed
        ff_dropout=0.1,  # Fixed
        mlp_hidden_mults=(4, 2),  # Fixed
        cont_embeddings='MLP',  # Fixed
        scalingfactor=10,  # Fixed
        attentiontype='colrow',  # Fixed
        final_mlp_style='sep',  # Fixed
        y_dim=2  # Fixed
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=params['lr'])

    scaler = MinMaxScaler()
    smote = SMOTE(random_state=42)
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    best_balance = -np.inf

    for train_index, val_index in kf.split(X_train):
        trainloader, valloader = setup_fold_data(train_index, val_index, X_train, y_train, scaler, smote, opt)

        for epoch in range(opt.epochs):
            model.train()
            for data in trainloader:
                optimizer.zero_grad()
                x_cont = data[0].to(device).float()
                y_gts = data[1].to(device)
                x_cont_enc = torch.cat([model.simple_MLP[i](x_cont[:, i:i + 1]) for i in range(model.num_continuous)],
                                       dim=-1)
                x_combined = model.fc(x_cont_enc)
                y_outs = model.mlpfory(x_combined)
                loss = torch.nn.functional.cross_entropy(y_outs, y_gts)
                loss.backward()
                optimizer.step()

        y_valid_fold, y_pred_fold = evaluate_model(model, valloader, device)
        _, _, balance, _ = calculate_metrics(y_valid_fold, y_pred_fold)

        if balance > best_balance:
            best_balance = balance

    print(f"Parameters: {params}, Balance: {best_balance}")
    return best_balance


if __name__ == '__main__':
    opt = get_args()
    print("Reading data...")
    data = pd.read_csv("EQ.csv")

    if 'class' not in data.columns:
        raise ValueError("데이터셋에 'class' 열이 없습니다. 열 이름을 확인하세요.")
    else:
        X = data.drop(['class'], axis=1)
        y = data['class']

    # 데이터를 train, test로 나눔
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    search_space = {
        'dim': [8, 16, 32],
        'depth': [3, 6, 9],
        'lr': [0.0001, 0.0005, 0.001]
    }

    print("Starting Ant Colony Optimization...")
    aco = AntColonyOptimizer(
        objective_function=objective_function,
        search_space=search_space,
        n_ants=10,
        n_best=5,
        n_iterations=20,
        decay=0.95,
        alpha=1,
        beta=1
    )

    best_params, best_score = aco.optimize()
    print(f"Best Parameters: {best_params}")
    print(f"Best Score: {best_score}")

    # 최종 테스트 데이터셋 평가
    print("Training final model with best parameters...")
    best_model = SAINT(
        num_continuous=len(X_train.columns),
        dim=best_params['dim'],
        depth=best_params['depth'],
        heads=8,  # Fixed
        attn_dropout=0.1,  # Fixed
        ff_dropout=0.1,  # Fixed
        mlp_hidden_mults=(4, 2),  # Fixed
        cont_embeddings='MLP',  # Fixed
        scalingfactor=10,  # Fixed
        attentiontype='colrow',  # Fixed
        final_mlp_style='sep',  # Fixed
        y_dim=2  # Fixed
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_model.to(device)
    optimizer = optim.AdamW(best_model.parameters(), lr=best_params['lr'])

    scaler = MinMaxScaler()
    smote = SMOTE(random_state=42)

    # Train with best parameters on the entire training set
    print("Normalizing and resampling training data...")
    X_train_normalized = scaler.fit_transform(X_train)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_normalized, y_train)
    X_train_tensor = torch.tensor(X_train_resampled).float()
    y_train_tensor = torch.tensor(y_train_resampled).long()
    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True, num_workers=4)

    for epoch in range(opt.epochs):
        print(f"Epoch {epoch + 1}/{opt.epochs}")
        best_model.train()
        for data in trainloader:
            optimizer.zero_grad()
            x_cont = data[0].to(device).float()
            y_gts = data[1].to(device)
            x_cont_enc = torch.cat([best_model.simple_MLP[i](x_cont[:, i:i + 1]) for i in range(best_model.num_continuous)],
                                   dim=-1)
            x_combined = best_model.fc(x_cont_enc)
            y_outs = best_model.mlpfory(x_combined)
            loss = torch.nn.functional.cross_entropy(y_outs, y_gts)
            loss.backward()
            optimizer.step()

    # 최종 테스트 데이터셋 평가
    print("Evaluating on test data...")
    X_test_normalized = scaler.transform(X_test)
    X_test_tensor = torch.tensor(X_test_normalized).float()
    y_test_tensor = torch.tensor(y_test.to_numpy()).long()

    print(f'Shape of X_test_tensor: {X_test_tensor.shape}')
    print(f'Shape of y_test_tensor: {y_test_tensor.shape}')

    if X_test_tensor.shape[0] != y_test_tensor.shape[0]:
        raise ValueError("Size mismatch between X_test_tensor and y_test_tensor")

    test_ds = TensorDataset(X_test_tensor, y_test_tensor)
    testloader = DataLoader(test_ds, batch_size=opt.batchsize, shuffle=False, num_workers=4)

    y_test_final, y_pred_final = evaluate_model(best_model, testloader, device)
    PD, PF, balance, FIR = calculate_metrics(y_test_final, y_pred_final)

    print('Test Set PD:', PD)
    print('Test Set PF:', PF)
    print('Test Set Balance:', balance)
    print('Test Set FIR:', FIR)

