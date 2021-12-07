from torch.nn.functional import one_hot
from custom_bert import BertForSequenceClassification
import torch
from captum.attr import visualization

class BertForSequenceClassificationExplanator:
    def __init__(self, bert_model: BertForSequenceClassification) -> None:
        self.bert_model = bert_model
        self.bert_model.eval()

    def generate_explanation(self, normalize_scores=True, get_logits=False, **input):
        output = self.bert_model(**input)
        logits = output.logits.reshape(input["input_ids"].shape[0], self.bert_model.config.num_labels)
        one_hot = torch.nn.functional.one_hot(torch.argmax(logits, dim=1), self.bert_model.config.num_labels)

        self.bert_model.zero_grad()
        backprop_me = (one_hot * logits).sum()
        backprop_me.backward(retain_graph=True)
        kwargs = {"alpha": 1}
        self.bert_model.relevance_propagation(one_hot.to(input["input_ids"].device), **kwargs)
        weighted_attention_relevance = None
        for attention_block in self.bert_model.bert.encoder.layer:
            rel = attention_block.attention.self.attention_relevance
            rel_grad = attention_block.attention.self.attention_grad
            weighted_layer_attention_relevance = torch.eye(rel.shape[-1]).to(rel.device) + (rel_grad * rel).clamp(min=0).mean(dim=1)
            if weighted_attention_relevance is None:
                weighted_attention_relevance = weighted_layer_attention_relevance.double()
            else:
                weighted_attention_relevance = torch.bmm(weighted_attention_relevance, weighted_layer_attention_relevance.double())
            assert not torch.isnan(weighted_attention_relevance).any()
        weighted_attention_relevance[:, 0, 0] = 0
        weighted_attention_relevance = weighted_attention_relevance[:, 0]
        if normalize_scores:
            weighted_attention_relevance = \
                (weighted_attention_relevance - weighted_attention_relevance.min()) / \
                (weighted_attention_relevance.max() - weighted_attention_relevance.min())
        if get_logits:
            return weighted_attention_relevance, one_hot, logits
        return weighted_attention_relevance, one_hot

    def vizualize(self, explanations, tokens, pred, label, label_string, label_being_explained=1):
        for i in range(explanations.shape[0]):
            # for j in range(0, explanations.shape[1]):
            #     if not(tokens[j] == "[PAD]") and not(tokens[j] == "[CLS]"):
            #         print((tokens[j], explanations[i, j].item()))
            mult = torch.max(explanations[i])
            print([(tokens[i][j], explanations[i, j].item()) for j in range(len(tokens[i].split()))])
            vis_data_records = [visualization.VisualizationDataRecord(
                                            ((1/mult) * explanations[i, :len(tokens[i].split())]).detach().numpy(),
                                            pred[i].item(),
                                            label_string[pred[i].item()],
                                            label_string[label[i].item()],
                                            label_string[label_being_explained],
                                            1,       
                                            tokens[i],
                                            1)]
            visualization.visualize_text(vis_data_records)

if __name__ == "__main__":
    # huggingface_model_name = "textattack/bert-base-uncased-SST-2"
    huggingface_model_name = "/home/sandor/Master/DD2412 Deep Learning, Advanced Course/Transformer-Explainability/BertForSequenceClassification"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    inputs = tokenizer(
        ["This movie was the best movie I have ever seen! some scenes were ridiculous, but acting was great.",
        "This movie is utter shit"], 
        return_tensors="pt", padding="max_length", truncation=True)
    # tokens = [tokenizer.convert_ids_to_tokens(inputs["input_ids"][i]) for i in range(inputs["input_ids"].shape[0])]
    tokens = ["This movie was the best movie I have ever seen! some scenes were ridiculous, but acting was great.",
        "This movie is utter shit"]

    model = BertForSequenceClassification.from_pretrained(huggingface_model_name, num_labels=2)
    print("Using activation func: ", model.config.hidden_act)
    explanator = BertForSequenceClassificationExplanator(model)
    exp, pred = explanator.generate_explanation(**inputs)
    explanator.vizualize(exp, tokens, torch.argmax(pred,dim=1), torch.tensor([1, 0]), ["NEG", "POS"])
    pass