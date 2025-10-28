import os
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List

import torch
from sklearn.model_selection import train_test_split
from transformers import (
	AutoTokenizer,
	AutoModelForSequenceClassification,
	Trainer,
	TrainingArguments,
)


@dataclass
class Constants:
	base_model: str = "distilbert-base-uncased"
	dataset_csv: str = "labeled_data.csv"  # updated to new dataset
	text_column: str = "tweet"  # column name in labeled_data.csv
	label_column: str = "class"  # 0=hate, 1=offensive, 2=neither -> map to binary toxic
	output_dir: str = os.path.join("model")
	max_length: int = 192
	test_size: float = 0.1
	random_state: int = 42


def prepare_dataframe(csv_path: str, text_col: str, label_col: str) -> pd.DataFrame:
	# Use latin-1 to accommodate dataset encoding quirks
	df = pd.read_csv(csv_path, encoding="latin-1")
	if text_col not in df.columns or label_col not in df.columns:
		raise ValueError(
			f"Expected columns '{text_col}' and '{label_col}' in {csv_path}. Found: {list(df.columns)}"
		)
	# Normalize labels:
	# - If labels are TRUE/FALSE strings -> map to 1/0
	# - If labels are {0,1,2} like Davidson et al. -> map to binary toxic (1 if != 2 else 0)
	labels_series = df[label_col]
	# Try boolean mapping first
	mapped_bool = labels_series.astype(str).str.upper().map({"TRUE": 1, "FALSE": 0})
	if mapped_bool.notna().any() and mapped_bool.isna().sum() < len(mapped_bool):
		labels_numeric = mapped_bool.fillna(labels_series)
	else:
		# Fallback: coerce to numeric
		labels_numeric = pd.to_numeric(labels_series, errors="coerce")
		unique_vals = set(int(v) for v in labels_numeric.dropna().unique().tolist())
		if unique_vals.issuperset({0,1,2}):
			# 0/1 toxic, 2 non-toxic -> binary
			labels_numeric = (labels_numeric != 2).astype(int)
		else:
			# Assume already 0/1
			labels_numeric = labels_numeric.astype(int)

	df[label_col] = labels_numeric
	df[text_col] = df[text_col].astype(str).fillna("")
	# Drop empty texts
	df = df[df[text_col].str.strip().ne("")].reset_index(drop=True)
	return df


class ToxicityDataset(torch.utils.data.Dataset):
	def __init__(self, texts: List[str], labels: List[int], tokenizer: AutoTokenizer, max_length: int):
		self.texts = texts
		self.labels = labels
		self.tokenizer = tokenizer
		self.max_length = max_length

	def __len__(self) -> int:
		return len(self.texts)

	def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
		item = self.tokenizer(
			self.texts[idx],
			padding="max_length",
			truncation=True,
			max_length=self.max_length,
			return_tensors="pt",
		)
		item = {k: v.squeeze(0) for k, v in item.items()}
		item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
		return item


def main():
	const = Constants()
	os.makedirs(const.output_dir, exist_ok=True)

	print("Loading dataset...")
	df = prepare_dataframe(const.dataset_csv, const.text_column, const.label_column)

	train_df, eval_df = train_test_split(
		df,
		test_size=const.test_size,
		random_state=const.random_state,
		stratify=df[const.label_column],
	)

	print("Loading tokenizer and model...")
	tokenizer = AutoTokenizer.from_pretrained(const.base_model)
	model = AutoModelForSequenceClassification.from_pretrained(const.base_model, num_labels=2)

	train_dataset = ToxicityDataset(
		texts=train_df[const.text_column].tolist(),
		labels=train_df[const.label_column].tolist(),
		tokenizer=tokenizer,
		max_length=const.max_length,
	)
	eval_dataset = ToxicityDataset(
		texts=eval_df[const.text_column].tolist(),
		labels=eval_df[const.label_column].tolist(),
		tokenizer=tokenizer,
		max_length=const.max_length,
	)

	print("Starting training...")
	training_args = TrainingArguments(
		output_dir=os.path.join(const.output_dir, "checkpoints"),
		eval_strategy="epoch",
		save_strategy="epoch",
		learning_rate=5e-5,
		per_device_train_batch_size=16,
		per_device_eval_batch_size=32,
		num_train_epochs=2,
		weight_decay=0.01,
		logging_steps=20,
		load_best_model_at_end=False,
		report_to=[],
	)

	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		tokenizer=tokenizer,
	)

	trainer.train()

	print(f"Saving model to {const.output_dir} ...")
	model.save_pretrained(const.output_dir)
	tokenizer.save_pretrained(const.output_dir)
	print("Done.")


if __name__ == "__main__":
	main()

