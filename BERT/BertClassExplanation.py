from torch.nn.functional import one_hot
from BERT.custom_bert import BertForSequenceClassification
import torch
from captum.attr import visualization

class BertForSequenceClassificationExplanator:
    def __init__(self, bert_model: BertForSequenceClassification) -> None:
        self.bert_model = bert_model
        self.bert_model.eval()

    def generate_explanation(self, normalize_scores=False, get_logits=False, **input):
        output = self.bert_model(**input)
        logits = output.logits.reshape(input["input_ids"].shape[0], self.bert_model.config.num_labels)
        one_hot = torch.nn.functional.one_hot(torch.argmax(logits, dim=1), self.bert_model.config.num_labels)

        self.bert_model.zero_grad()
        backprop_me = (one_hot * logits).sum()
        backprop_me.backward(retain_graph=True)
        kwargs = {"alpha": 1}
        self.bert_model.relprop(one_hot.to(input["input_ids"].device), **kwargs)
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
            print([(tokens[i][j], explanations[i, j].item()) for j in range(len(tokens[i]))])
            vis_data_records = [visualization.VisualizationDataRecord(
                                            explanations[i].detach().numpy(),
                                            pred[i].item(),
                                            label_string[pred[i].item()],
                                            label_string[label[i].item()],
                                            label_string[label_being_explained],
                                            1,       
                                            tokens[i],
                                            1)]
            visualization.visualize_text(vis_data_records)

if __name__ == "__main__":
    huggingface_model_name = "textattack/bert-base-uncased-SST-2"
    # huggingface_model_name = "/home/sandor/Master/DD2412 Deep Learning, Advanced Course/Transformer-Explainability/BertForSequenceClassification"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    inputs = tokenizer(
        ["This movie was the best movie I have ever seen! some scenes were ridiculous, but acting was great."],
        return_tensors="pt", padding="max_length", truncation=True)
    tokens = [tokenizer.convert_ids_to_tokens(inputs["input_ids"][i]) for i in range(inputs["input_ids"].shape[0])]
    del tokenizer
    # tokens = ["This movie was the best movie I have ever seen! some scenes were ridiculous, but acting was great.", "This movie is utter shit"]

    model = BertForSequenceClassification.from_pretrained(huggingface_model_name, num_labels=2)
    print("Using activation func: ", model.config.hidden_act)
    explanator = BertForSequenceClassificationExplanator(model)
    exp, pred = explanator.generate_explanation(**inputs)
    explanator.vizualize(exp, tokens, torch.argmax(pred,dim=1), torch.tensor([1, 0]), ["NEG", "POS"])
    # Notebook example
    # [('[CLS]', 0.0), ('this', 0.4398406744003296), ('movie', 0.3385171890258789), ('was', 0.2850261628627777), ('the', 0.3722951412200928), ('best', 0.6413642764091492), ('movie', 0.3098682463169098), ('i', 0.20284101366996765), ('have', 0.12214731425046921), ('ever', 0.15835356712341309), ('seen', 0.2082878053188324), ('!', 0.6001579761505127), ('some', 0.021879158914089203), ('scenes', 0.05488050356507301), ('were', 0.0371897891163826), ('ridiculous', 0.03780526667833328), (',', 0.02076297625899315), ('but', 0.44531309604644775), ('acting', 0.45006945729255676), ('was', 0.5168584585189819), ('great', 1.0), ('.', 0.035734280943870544), ('[SEP]', 0.10382220149040222)]