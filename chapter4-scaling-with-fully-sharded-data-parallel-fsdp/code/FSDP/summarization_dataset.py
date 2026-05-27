import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class wikihow(Dataset):
    def __init__(self, tokenizer, type_path, num_samples, input_length, output_length, print_text=False, data_dir='data/'):
        """
        WikiHow dataset loader that reads from local CSV files.
        
        Args:
            tokenizer: Tokenizer to use
            type_path: Dataset split ('train' or 'validation')
            num_samples: Number of samples to use (None for all)
            input_length: Maximum input sequence length
            output_length: Maximum output sequence length
            print_text: Whether to print text samples
            data_dir: Directory containing the CSV files
        """
        # Load from local CSV files
        csv_path = os.path.join(data_dir, 'wikihowAll.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset file not found: {csv_path}. Please run download_dataset.sh first.")
        
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Split into train/validation (80/20 split)
        # Use a fixed random seed for reproducibility
        np.random.seed(42)
        shuffled_indices = np.random.permutation(len(df))
        split_idx = int(0.8 * len(df))
        
        if type_path == 'train':
            indices = shuffled_indices[:split_idx]
        elif type_path == 'validation':
            indices = shuffled_indices[split_idx:]
        else:
            raise ValueError(f"type_path must be 'train' or 'validation', got {type_path}")
        
        # Select subset
        self.df = df.iloc[indices].reset_index(drop=True)
        
        # Limit number of samples if specified
        if num_samples and num_samples < len(self.df):
            self.df = self.df.head(num_samples).reset_index(drop=True)
        
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.output_length = output_length
        self.print_text = print_text

    def __len__(self):
        return len(self.df)

    def clean_text(self, text):
        text = text.replace('Example of text:', '')
        text = text.replace('Example of Summary:', '')
        text = text.replace('\n','')
        text = text.replace('``', '')
        text = text.replace('"', '')

        return text


    def convert_to_features(self, example_batch):
        # Tokenize contexts and questions (as pairs of inputs)

        if self.print_text:
            print("Input Text: ", self.clean_text(example_batch['text']))
#         input_ = self.clean_text(example_batch['text']) + " </s>"
#         target_ = self.clean_text(example_batch['headline']) + " </s>"

        input_ = self.clean_text(example_batch['text'])
        target_ = self.clean_text(example_batch['headline'])

        source = self.tokenizer.batch_encode_plus([input_], max_length=self.input_length,
                                                     padding='max_length', truncation=True, return_tensors="pt")

        targets = self.tokenizer.batch_encode_plus([target_], max_length=self.output_length,
                                                     padding='max_length', truncation=True, return_tensors="pt")


        return source, targets

    def __getitem__(self, index):
        # Get row from dataframe
        row = self.df.iloc[index]
        # Handle NaN values - use pandas .get() with default or direct access
        text = str(row['text']) if 'text' in self.df.columns and pd.notna(row['text']) else ''
        headline = str(row['headline']) if 'headline' in self.df.columns and pd.notna(row['headline']) else ''
        example = {
            'text': text,
            'headline': headline
        }
        source, targets = self.convert_to_features(example)

        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask    = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

def get_dataset(tokenizer, type_path, num_samples, args, input_length=512, output_length=150):
    """
    Helper function to create a wikihow dataset.
    
    Args:
        tokenizer: Tokenizer to use
        type_path: Dataset split ('train' or 'validation')
        num_samples: Number of samples to use
        args: Arguments object (currently unused, kept for compatibility)
        input_length: Maximum input sequence length (default: 512)
        output_length: Maximum output sequence length (default: 150)
    
    Returns:
        wikihow dataset instance
    """
    return wikihow(
        tokenizer=tokenizer,
        type_path=type_path,
        num_samples=num_samples,
        input_length=input_length,
        output_length=output_length
    )
