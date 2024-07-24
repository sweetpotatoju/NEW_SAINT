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
    parser.add_argument('--cont_embeddings', default='MLP', type=str, choices=['MLP', 'Noemb', 'pos_singleMLP'])
    parser.add_argument('--embedding_size', default=32, type=int)
    parser.add_argument('--transformer_depth', default=6, type=int)
    parser.add_argument('--attention_heads', default=8, type=int)
    parser.add_argument('--attention_dropout', default=0.1, type=float)
    parser.add_argument('--ff_dropout', default=0.1, type=float)
    parser.add_argument('--attentiontype', default='colrow', type=str, choices=['col', 'colrow', 'row', 'justmlp', 'attn', 'attnmlp'])
    parser.add_argument('--optimizer', default='AdamW', type=str, choices=['AdamW', 'Adam', 'SGD'])
    parser.add_argument('--scheduler', default='cosine', type=str, choices=['cosine', 'linear'])
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batchsize', default=64, type=int)
    parser.add_argument('--savemodelroot', default='./bestmodels', type=str)
    parser.add_argument('--run_name', default='testrun', type=str)
    parser.add_argument('--set_seed', default=1, type=int)
    parser.add_argument('--dset_seed', default=5, type=int)
    parser.add_argument('--active_log', action='store_true')
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--pretrain_epochs', default=1, type=int)
    parser.add_argument('--pt_tasks', default=['contrastive', 'denoising'], type=str, nargs='*', choices=['contrastive', 'contrastive_sim', 'denoising'])
    parser.add_argument('--pt_aug', default=[], type=str, nargs='*', choices=['mixup', 'cutmix'])
    parser.add_argument('--pt_aug_lam', default=0.1, type=float)
    parser.add_argument('--mixup_lam', default=0.3, type=float)
    parser.add_argument('--train_mask_prob', default=0, type=float)
    parser.add_argument('--mask_prob', default=0, type=float)
    parser.add_argument('--ssl_avail_y', default=0, type=int)
    parser.add_argument('--pt_projhead_style', default='diff', type=str, choices=['diff', 'same', 'nohead'])
    parser.add_argument('--nce_temp', default=0.7, type=float)
    parser.add_argument('--lam0', default=0.5, type=float)
    parser.add_argument('--lam1', default=10, type=float)
    parser.add_argument('--lam2', default=1, type=float)
    parser.add_argument('--lam3', default=10, type=float)
    parser.add_argument('--final_mlp_style', default='sep', type=str, choices=['common', 'sep'])
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


if __name__ == '__main__':
    opt = get_args()
    data = pd.read_csv("EQ.csv")

    if 'class' not in data.columns:
        print("데이터셋에 'class' 열이 없습니다. 열 이름을 확인하세요.")
    else:
        X = data.drop(['class'], axis=1)
        y = data['class']

    # 데이터를 train, test로 나눔
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    smote = SMOTE(random_state=42)
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    pd_list = []
    pf_list = []
    bal_list = []
    fir_list = []
    best_model = None
    best_balance = -np.inf  # Initialize to a very low value

    for train_index, val_index in kf.split(X_train):
        trainloader, valloader = setup_fold_data(train_index, val_index, X_train, y_train, scaler, smote, opt)

        model = SAINT(
            num_continuous=len(X_train.columns),
            dim=8,
            depth=int(opt.transformer_depth),
            heads=int(opt.attention_heads),
            attn_dropout=opt.attention_dropout,
            ff_dropout=opt.ff_dropout,
            mlp_hidden_mults=(4, 2),
            cont_embeddings=opt.cont_embeddings,
            scalingfactor=10,
            attentiontype=opt.attentiontype,
            final_mlp_style=opt.final_mlp_style,
            y_dim=2
        )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=opt.lr)

        for epoch in range(opt.epochs):
            start_time = time.time()
            model.train()
            running_train_loss = 0.0
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
                running_train_loss += loss.item()

            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for data in valloader:
                    x_cont = data[0].to(device).float()
                    y_gts = data[1].to(device)
                    x_cont_enc = torch.cat(
                        [model.simple_MLP[i](x_cont[:, i:i + 1]) for i in range(model.num_continuous)], dim=-1)
                    x_combined = model.fc(x_cont_enc)
                    y_outs = model.mlpfory(x_combined)
                    loss = torch.nn.functional.cross_entropy(y_outs, y_gts)
                    running_val_loss += loss.item()

            epoch_time = time.time() - start_time
            if epoch % 1 == 0:
                print(
                    f"Epoch [{epoch}/{opt.epochs}], Train Loss: {running_train_loss:.4f}, Val Loss: {running_val_loss:.4f}, Time: {epoch_time:.2f}s")

        y_valid_fold, y_pred_fold = evaluate_model(model, valloader, device)
        PD, PF, balance, FIR = calculate_metrics(y_valid_fold, y_pred_fold)
        pd_list.append(PD)
        pf_list.append(PF)
        bal_list.append(balance)
        fir_list.append(FIR)

        # Save the best model based on balance metric
        if balance > best_balance:
            best_balance = balance
            best_model = model

    avg_PD = np.mean(pd_list)
    avg_PF = np.mean(pf_list)
    avg_balance = np.mean(bal_list)
    avg_FIR = np.mean(fir_list)

    print('Average PD:', avg_PD)
    print('Average PF:', avg_PF)
    print('Average Balance:', avg_balance)
    print('Average FIR:', avg_FIR)

    # 최종 테스트 데이터셋 평가
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
