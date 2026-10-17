# -*- coding: utf-8 -*-
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
import random
import argparse
import numpy as np
import os
import json

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(3407)

# 4PL IRT Model with Embedding Output


class IRT(nn.Module):
    def __init__(self, model_type, num_students, num_items, hidden_dim=64, embedding_dim=128):
        super().__init__()
        self.model_type = model_type
        self.num_students = num_students
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # Student ability network
        self.student_net = nn.Sequential(
            nn.Linear(num_students, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.student_ability_out = nn.Linear(embedding_dim, 1)

        # Item parameter network
        self.item_net = nn.Sequential(
            nn.Linear(num_items, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.item_param_out = nn.Linear(embedding_dim, int(self.model_type[0]))

    def forward(self, student, item):
        student_embedding = self.student_net(student)
        item_embedding = self.item_net(item)

        theta = self.student_ability_out(student_embedding)
        params = self.item_param_out(item_embedding)

        # Parameter parsing
        beta = params[:, 0].unsqueeze(1)    # diff

        if self.model_type == "1pl":
            # P = sigmoid(theta - beta)
            logits = theta - beta
            prob = torch.sigmoid(logits)
 
        elif self.model_type == "2pl":
            # P = sigmoid(a * (theta - beta))
            a = params[:, 1].unsqueeze(1)   # disc
            logits = a * (theta - beta)
            prob = torch.sigmoid(logits)
 
        elif self.model_type == "3pl":
            # P = c + (1 - c) * sigmoid(a * (theta - beta))
            a = params[:, 1].unsqueeze(1)
            c = torch.sigmoid(params[:, 2].unsqueeze(1))    # guessing
            logits = a * (theta - beta)
            prob = c + (1 - c) * torch.sigmoid(logits)
 
        elif self.model_type == "4pl":
            # P = c + (d - c) * sigmoid(a * (theta - beta))
            a = params[:, 1].unsqueeze(1)
            c = torch.sigmoid(params[:, 2].unsqueeze(1))
            d = torch.sigmoid(params[:, 3].unsqueeze(1))    # upper xxx
            logits = a * (theta - beta)
            prob = c + (d - c) * torch.sigmoid(logits)

        return prob, theta, params, student_embedding, item_embedding

# Dataset class


class IRTDataset(Dataset):
    def __init__(self, df):
        self.num_students = len(df)
        self.num_items = len(df.columns)
        self.data = []
        for sid in range(self.num_students):
            for qid in range(self.num_items):
                label = df.iloc[sid, qid]
                if not pd.isnull(label):
                    self.data.append((sid, qid, int(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sid, qid, label = self.data[idx]
        student_vec = torch.zeros(self.num_students).float()
        student_vec[sid] = 1.0
        item_vec = torch.zeros(self.num_items).float()
        item_vec[qid] = 1.0
        return {
            "student": student_vec,
            "item": item_vec,
            "label": torch.tensor(label, dtype=torch.float32),
        }

# Main program


def main(args):
    # check the starting point
    out_folder = os.path.join(args.output_dir, args.benchmark, args.model_type, f'N={args.n_students}')
    os.makedirs(out_folder, exist_ok=True)
    start_idx = len([f for f in os.listdir(out_folder) if 'est_' in f]) + 1
    if start_idx == 51:
        print(f'{args.model_type} on {args.benchmark} with {args.n_students} models has been finished!')
        return
    else:
        print(f"Fitting {args.model_type} on {args.benchmark} with {args.n_students} models, device={device}")

    # Load data
    for i in trange(start_idx, 51, desc='rep', position=0, leave=True):
        input_path = os.path.join(args.input_dir, args.benchmark, args.model_type, f'resp_sim_rep{i}.csv')
        train_df = pd.read_csv(input_path, header=0, index_col=0, nrows=args.n_students)

        num_students = len(train_df)
        num_items = len(train_df.columns)

        if i == start_idx:
            print(f"Number of students: {num_students}, Number of items: {num_items}")

        # Create dataset and data loader
        train_dataset = IRTDataset(train_df)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True)

        # Model initialization
        model = IRT(
            model_type=args.model_type,
            num_students=num_students,
            num_items=num_items,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.embedding_dim
        ).to(device)

        # Optimizer and loss function
        optimizer = optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        criterion = nn.BCELoss()

        for epoch in range(args.max_epochs):
            model.train()
            total_loss = 0.0

            for batch in train_loader:
                optimizer.zero_grad()
                student = batch["student"].to(device)
                item = batch["item"].to(device)
                labels = batch["label"].to(device)
                prob, _, _, _, _ = model(student, item)
                loss = criterion(prob.squeeze(), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{args.max_epochs}, Loss: {avg_loss:.4f}", flush=True)

        # Save parameters
        save_params(args, i, model, train_df)

# Parameter saving function


def save_params(args, idx, model, train_df):
    num_students = len(train_df)
    num_items = len(train_df.columns)

    # Get student abilities
    student_abilities = torch.zeros(num_students, 1)
    with torch.no_grad():
        for sid in range(num_students):
            student_vec = torch.zeros(
                num_students).float().unsqueeze(0).to(device)
            student_vec[0, sid] = 1.0
            embedding = model.student_net(student_vec)
            ability = model.student_ability_out(embedding)
            student_abilities[sid] = ability.cpu()

    # Get item parameters
    col_map = {
        "1pl": ["difficulty"],
        "2pl": ["difficulty", "discrimination"],
        "3pl": ["difficulty", "discrimination", "guessing"],
        "4pl": ["difficulty", "discrimination", "guessing", "feasibility"],
    }
    item_params = torch.zeros(num_items, int(args.model_type[0]))
    with torch.no_grad():
        for qid in range(num_items):
            item_vec = torch.zeros(num_items).float().unsqueeze(0).to(device)
            item_vec[0, qid] = 1.0
            embedding = model.item_net(item_vec)
            params = model.item_param_out(embedding)
            item_params[qid] = params.cpu()
    item_param_df = pd.DataFrame(item_params.numpy(), columns=col_map[args.model_type])
    if 'guessing' in item_param_df.columns:
        item_param_df["guessing"] = 1 / (1 + np.exp(-item_param_df["guessing"]))
    if 'feasibility' in item_param_df.columns:
        item_param_df["feasibility"] = 1 / (1 + np.exp(-item_param_df["feasibility"]))
    res = item_param_df.to_dict(orient='list')
    res['ability'] = student_abilities.squeeze().numpy().tolist()

    out_path = os.path.join(args.output_dir, args.benchmark, args.model_type, f'N={args.n_students}/est_rep{idx}.json')
    with open(out_path, 'w') as of:
        json.dump(res, of, indent=4)
    print("Item parameters saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['1pl', '2pl', '3pl', '4pl'])
    parser.add_argument('--n_students', type=int, choices=[30, 100, 180, 400, 1000])
    parser.add_argument('--benchmark', type=str, choices=['arc', 'gsm8k', 'hellaswag', 'mmlu', 'truthfulqa', 'winogrande'])
    parser.add_argument('--input_dir', type=str, default='/path/to/input/data')
    parser.add_argument('--output_dir', type=str, default='/path/to/output/folder')
    parser.add_argument('--learning_rate', type=float,
                        default=0.003, help='Learning rate')
    parser.add_argument('--weight_decay', type=float,
                        default=0.0001, help='Weight decay')
    parser.add_argument('--batch_size', type=int,
                        default=1024, help='Batch size')
    parser.add_argument('--max_epochs', type=int, 
                        default=30, help='Maximum number of epochs')
    parser.add_argument('--hidden_dim', type=int,
                        default=64, help='Hidden dimension')
    parser.add_argument('--embedding_dim', type=int, 
                        default=128, help='Embedding dimension for students and items')
    args = parser.parse_args()
    main(args)
