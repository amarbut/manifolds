import argparse
import torch
import torch.nn as nn
from transformers import AutoModel, BertPreTrainedModel, BertConfig, BertForSequenceClassification

class ExtendedBERT (BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = AutoModel.from_pretrained(self.config.model_name_or_path)
        self.classifier = nn.ModuleDict(
            {f"fc{i}":nn.Linear(self.config.hidden_size, self.config.intermediate_size) for i in range(self.config.num_fc)})
        self.post_init()

def build_bert(model_name_or_path, num_fc):
	config = BertConfig.from_pretrained(model_name_or_path)
	config.model_name_or_path = model_name_or_path
	config.num_fc = num_fc
	model = ExtendedBERT(config)
	file_name = f"{model_name_or_path}_{num_fc}layers"
	model.save_pretrained(file_name)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_name_or_path', help = 'Name or file location of pre-trained Bert model', default = 'bert-base-uncased', required = False)
	parser.add_argument('--num_fc', help = 'Number of fully connected layers to add to pre-trained model', type = int, default = 0, required = False)

	args = vars(parser.parse_args())

	build_bert(**args)
