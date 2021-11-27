from torch.nn.functional import one_hot
from custom_bert import BertForSequenceClassification
import torch
from captum.attr import visualization

class BertForSequenceClassificationExplanator:
    def __init__(self, bert_model: BertForSequenceClassification) -> None:
        self.bert_model = bert_model

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
        normalized_class_score = torch.nn.functional.softmax(logits, dim=1)
        preds = torch.argmax(normalized_class_score, dim=1).float().requires_grad_(True)

        self.bert_model.zero_grad()
        preds.backward(retain_graph=True)
        one_hot_preds = torch.zeros((preds.shape[0], self.bert_model.config.num_labels))
        one_hot_preds[:, preds.long()] = 1
        kwargs = {"alpha": 1}
        self.bert_model.relevance_propagation(one_hot_preds.to(input["input_ids"].device), **kwargs)
        relevances = []
        rel_prod = None
        for attention_block in self.bert_model.bert.encoder.layer:
            rel = attention_block.attention.self.attention_relevance.clamp(min=0).mean(dim=1)
            relevances.append(rel)
            rel = torch.eye(rel.shape[1]) + rel
            rel = (rel / rel.sum(dim=-1, keepdim=True).clamp(min=1e-9)).clamp(max=1e20)
            if rel_prod is None:
                rel_prod = rel
            else:
                rel_prod = torch.bmm(rel_prod, rel)
            assert not torch.isnan(rel_prod).any()
        final_rel = self.compute_rollout_attention(relevances)
        final_rel[:, 0, 0] = 0
        rel_prod[:, 0, 0] = 0
        # return final_rel[:, 0], normalized_class_score
        assert torch.isclose(final_rel, rel_prod, atol=1e-4).all()
        return rel_prod[:, 0], normalized_class_score

    def vizualize(self, explanations, tokens, pred_prob, true_class):
        for i in range(explanations.shape[0]):
            for j in range(1, len(tokens)):
                if not(tokens[j] == "[PAD]") and not(tokens[j] == "[CLS]"):
                    print((tokens[j], explanations[i, j].item()))
            vis_data_records = [visualization.VisualizationDataRecord(
                                            explanations.tolist(),
                                            torch.argmax(pred_prob, dim=1).tolist(),
                                            true_class,
                                            true_class,
                                            true_class,
                                            1,       
                                            tokens,
                                            1)]
            visualization.visualize_text(vis_data_records)
        pass

if __name__ == "__main__":
    huggingface_model_name = "textattack/bert-base-uncased-SST-2"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(huggingface_model_name)

    inputs = tokenizer(
        "This movie was the best movie I have ever seen! some scenes were ridiculous, but acting was great.", 
        return_tensors="pt", padding="max_length", truncation=True)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].flatten())

    model = BertForSequenceClassification.from_pretrained(huggingface_model_name, num_labels=2)
    print("Using activation func: ", model.config.hidden_act)
    explanator = BertForSequenceClassificationExplanator(model)
    exp, pred_prob = explanator.generate_explanation(**inputs)
    explanator.vizualize(exp, tokens, pred_prob, 1)
    # outputs = model(**inputs)
    # class_score = torch.nn.functional.softmax(outputs.logits, dim=1)
    # class_preds = torch.argmax(class_score, dim=1)
    # rel = model.relevance_propagation(class_preds, alpha=1)

    print("Done!")