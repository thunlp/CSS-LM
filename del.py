#from transformers import BertModel, RobertaModel

#q = BertModel.from_pretrained('bert-base-uncased')
#b = RobertaModel.from_pretrained('roberta-base')

from transformers import BertConfig, BertModel
model = BertModel.from_pretrained("bert-base-uncased")
