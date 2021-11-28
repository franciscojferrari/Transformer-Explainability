from torch.nn.functional import one_hot
from custom_bert import BertForSequenceClassification
import torch
from captum.attr import visualization

class BertForSequenceClassificationExplanator:
    def __init__(self, bert_model: BertForSequenceClassification) -> None:
        self.bert_model = bert_model
        self.bert_model.eval()

    def compute_rollout_attention(self, all_layer_matrices, start_layer=0):
        # adding residual consideration
        num_tokens = all_layer_matrices[0].shape[1]
        batch_size = all_layer_matrices[0].shape[0]
        eye = torch.eye(num_tokens).expand(
            batch_size, num_tokens, num_tokens).to(
            all_layer_matrices[0].device)
        all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
        # all_layer_matrices = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
        #                       for i in range(len(all_layer_matrices))]
        joint_attention = all_layer_matrices[start_layer]
        for i in range(start_layer+1, len(all_layer_matrices)):
            joint_attention = all_layer_matrices[i].bmm(joint_attention)
        return joint_attention

    def generate_explanation(self, **input):
        output = self.bert_model(**input)
        logits = output.logits.reshape(input["input_ids"].shape[0], self.bert_model.config.num_labels)
        one_hot = torch.nn.functional.one_hot(torch.argmax(logits, dim=1))

        self.bert_model.zero_grad()
        (one_hot * logits).sum().backward(retain_graph=True)
        kwargs = {"alpha": 1}
        self.bert_model.relevance_propagation(one_hot.to(input["input_ids"].device), **kwargs)
        weighted_attention_relevance = None
        for attention_block in self.bert_model.bert.encoder.layer:
            rel = attention_block.attention.self.attention_relevance
            rel_grad = attention_block.attention.self.attention_grad
            weighted_layer_attention_relevance = torch.eye(rel.shape[-1]) + (rel_grad * rel).clamp(min=0).mean(dim=1)
            if weighted_attention_relevance is None:
                weighted_attention_relevance = weighted_layer_attention_relevance.double()
            else:
                weighted_attention_relevance = torch.bmm(weighted_attention_relevance, weighted_layer_attention_relevance.double())
            assert not torch.isnan(weighted_attention_relevance).any()
        weighted_attention_relevance[:, 0, 0] = 0
        return weighted_attention_relevance[:, 0], one_hot

    def vizualize(self, explanations, tokens, pred, label, label_string, label_being_explained=1):
        for i in range(explanations.shape[0]):
            # for j in range(0, explanations.shape[1]):
            #     if not(tokens[j] == "[PAD]") and not(tokens[j] == "[CLS]"):
            #         print((tokens[j], explanations[i, j].item()))
            vis_data_records = [visualization.VisualizationDataRecord(
                                            explanations[i, :len(tokens)].detach().numpy(),
                                            pred[i].item(),
                                            label_string[pred[i].item()],
                                            label_string[label[i].item()],
                                            label_string[label_being_explained],
                                            explanations[i].sum(),       
                                            tokens[i],
                                            1)]
            visualization.visualize_text(vis_data_records)
            
        pass

if __name__ == "__main__":
    huggingface_model_name = "textattack/bert-base-uncased-SST-2"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(huggingface_model_name)

    inputs = tokenizer(
        ["This movie was the best movie I have ever seen! some scenes were ridiculous, but acting was great.",
        "Pretty gud moive",
        "This movie is utter shit"], 
        return_tensors="pt", padding="max_length", truncation=True)
    tokens = [tokenizer.convert_ids_to_tokens(inputs["input_ids"][i]) for i in range(inputs["input_ids"].shape[0])]

    model = BertForSequenceClassification.from_pretrained(huggingface_model_name, num_labels=2)
    print("Using activation func: ", model.config.hidden_act)
    explanator = BertForSequenceClassificationExplanator(model)
    exp, pred = explanator.generate_explanation(**inputs)
    explanator.vizualize(exp, tokens, torch.argmax(pred,dim=1), torch.tensor([1, 1, 0]), ["NEG", "POS"])
    # outputs = model(**inputs)
    # class_score = torch.nn.functional.softmax(outputs.logits, dim=1)
    # class_preds = torch.argmax(class_score, dim=1)
    # rel = model.relevance_propagation(class_preds, alpha=1)

    print("Done!")