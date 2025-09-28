# -*- coding:utf-8 -*-

import os
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule


class CTRLDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)


class DataModule(LightningDataModule):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        if self.cfg.datamodule.num_attr == 1:
            self.data = self.setup()
        self.col_fn = self.collate_fn

    def get_data(self, path):
        result, cur = [], []
        with open(path, "r") as f:
            for line in f:
                s = line.strip()
                if s == "<EOD>":
                    if cur:
                        result.append(cur)
                        cur = []
                else:
                    cur.append(s)
        return result

    def get_dial_data(self, dial_list, att_list, label2id, mode="emo", balanced=False):
        out = []
        for dial, attrs in zip(dial_list, att_list):
            if len(dial) != len(attrs):
                raise ValueError("dialog and attribute length are mismatch")

            cur_dial = ""
            for utt, label in zip(dial, attrs):
                att_label = int(label) - 1
                cur_dial += utt + self.tokenizer.eos_token

                # 첫 발화 혹은 no-emotion(-1) 스킵
                if cur_dial == utt + self.tokenizer.eos_token or att_label == -1:
                    continue

                inputs = self.tokenizer.bos_token + cur_dial
                response = utt + self.tokenizer.eos_token
                if len(self.tokenizer.encode(inputs)) > 512:
                    continue

                out.append(
                    {
                        "inputs": inputs,
                        "response": response,
                        "dialog_history": inputs[: -len(response)],
                        "label": att_label,
                    }
                )
        return out

    def setup(self, stage=None):
        output = {"train": [], "valid": [], "test": []}

        dailydial = {"train": {}, "valid": {}, "test": {}}
        for split in ["train", "valid", "test"]:
            for kind in ["dial", "emo", "act"]:
                path = os.path.join(self.cfg.data_dir, f"parse_{split}", f"{kind}.txt")
                dailydial[split][kind] = self.get_data(path)

        if "emo" in self.cfg.datamodule.data_name:
            label2id_emo = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
            output["train"] = self.get_dial_data(
                dailydial["train"]["dial"], dailydial["train"][self.cfg.datamodule.data_name], label2id_emo, mode="emo"
            )
            output["valid"] = self.get_dial_data(
                dailydial["valid"]["dial"], dailydial["valid"][self.cfg.datamodule.data_name], label2id_emo, mode="emo"
            )
            output["test"] = self.get_dial_data(
                dailydial["test"]["dial"], dailydial["test"][self.cfg.datamodule.data_name], label2id_emo, mode="emo"
            )

        elif "act" in self.cfg.datamodule.data_name:
            label2id_act = ["inform", "question", "directive", "commissive"]
            output["train"] = self.get_dial_data(
                dailydial["train"]["dial"], dailydial["train"][self.cfg.datamodule.data_name], label2id_act, mode="emo"
            )
            output["valid"] = self.get_dial_data(
                dailydial["valid"]["dial"], dailydial["valid"][self.cfg.datamodule.data_name], label2id_act, mode="emo"
            )
            output["test"] = self.get_dial_data(
                dailydial["test"]["dial"], dailydial["test"][self.cfg.datamodule.data_name], label2id_act, mode="emo"
            )

        elif "multi" in self.cfg.datamodule.data_name:
            label2id_emo = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
            label2id_act = ["inform", "question", "directive", "commissive"]
            labeled_dial = [[[] for _ in label2id_act] for _ in label2id_emo]

            for dial, emo_list, act_list in zip(dailydial["test"]["dial"], dailydial["test"]["emo"], dailydial["test"]["act"]):
                cur_dial = ""
                for utt, emo, act in zip(dial, emo_list, act_list):
                    emo_label = int(emo) - 1
                    act_label = int(act) - 1
                    cur_dial += utt + self.tokenizer.eos_token
                    if emo_label == -1:
                        continue

                    attribute_value = (
                        self.tokenizer.bos_token
                        + label2id_emo[emo_label]
                        + self.tokenizer.eos_token
                        + label2id_act[act_label]
                        + self.tokenizer.eos_token
                    )
                    inputs = attribute_value + cur_dial
                    response = utt + self.tokenizer.eos_token
                    if len(self.tokenizer.encode(inputs)) > 512:
                        continue

                    item = {
                        "attribute_value": attribute_value,
                        "inputs": inputs,
                        "response": response,
                        "dialog_history": inputs[: -len(response)],
                        "label": (emo_label, act_label),
                    }
                    labeled_dial[emo_label][act_label].append(item)
                    output["train"].append(item)

        return output

    def collate_fn(self, batch):
        batch_src = [b["inputs"] for b in batch]
        batch_hist = [b["dialog_history"] for b in batch]
        batch_lbl = [b["label"] for b in batch]

        model_input = self.tokenizer(
            batch_src,
            max_length=self.cfg.datamodule.max_seq_length,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
            add_special_tokens=False,
            return_attention_mask=True,
        )
        history_input = self.tokenizer(
            batch_hist,
            return_tensors="pt",
            max_length=self.cfg.datamodule.max_seq_length,
            padding="max_length",
            add_special_tokens=False,
        )

        input_ids, attention_mask = model_input["input_ids"], model_input["attention_mask"]
        history_ids = history_input["input_ids"]
        label_ids = input_ids - history_ids
        label_ids += self.tokenizer.pad_token_id
        label_ids = label_ids.masked_fill(label_ids == self.tokenizer.pad_token_id, -100)
        input_ids[input_ids == self.tokenizer.pad_token_id] = self.tokenizer.eos_token_id

        device = input_ids.device
        dtype_str = getattr(self.cfg.learner, "class_labels_dtype", "float32")
        dtype = getattr(torch, dtype_str)

        # 응답 토큰 인덱스들 (vocab index)
        label_idx = torch.masked_select(label_ids, label_ids != -100).to(device=device, dtype=torch.long)

        # 각 샘플별 응답 길이
        res_mask = (label_ids != -100).to(torch.long)
        res_len = res_mask.sum(dim=1)  # [B]
        expand_label = torch.repeat_interleave(torch.tensor(batch_lbl, device=device), res_len).to(torch.long)

        N = label_idx.numel()                   # 총 응답 토큰 수
        V = len(self.tokenizer)                 # vocab size
        C = int(self.cfg.learner.num_labels)    # 클래스 수

        # (N, C, V) 기본값 1/C로 채움
        class_labels = torch.full((N, C, V), 1.0 / C, dtype=dtype, device=device)

        # 모든 클래스에 대해, 해당 토큰 위치(vocab index)만 0으로
        rows = torch.arange(N, device=device).unsqueeze(1).expand(N, C)     # [N, C]
        classes = torch.arange(C, device=device).unsqueeze(0).expand(N, C)  # [N, C]
        cols = label_idx.unsqueeze(1).expand(N, C)                           # [N, C]
        class_labels[rows, classes, cols] = 0.0

        # 정답 클래스 위치만 1로
        class_labels[torch.arange(N, device=device), expand_label, label_idx] = 1.0

        # (N, C*V)로 reshape
        class_labels = class_labels.view(N, C * V)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_ids": label_ids,
            "labels": torch.tensor(batch_lbl, device=device),
            "class_labels": class_labels,
        }

    def collate_fn_test(self, batch):
        batch_hist = [b["dialog_history"] for b in batch]
        batch_lbl = [b["label"] for b in batch]

        model_input = self.tokenizer(
            batch_hist,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=False,
            return_attention_mask=True,
        )

        return {
            "input_ids": model_input["input_ids"],
            "attention_mask": model_input["attention_mask"],
            "labels": torch.tensor(batch_lbl),
        }

    def train_dataloader(self):
        return DataLoader(
            CTRLDataset(self.data["train"]),
            batch_size=self.cfg.datamodule.batch_size,
            shuffle=True,
            collate_fn=self.col_fn,
            num_workers=30,
        )

    def val_dataloader(self):
        return DataLoader(
            CTRLDataset(self.data["valid"]),
            batch_size=self.cfg.datamodule.batch_size,
            shuffle=False,
            collate_fn=self.col_fn,
            num_workers=30,
        )

    def test_dataloader(self):
        return DataLoader(
            CTRLDataset(self.data["test"]),
            batch_size=1,
            shuffle=False,
            collate_fn=self.collate_fn_test,
            num_workers=1,
        )
